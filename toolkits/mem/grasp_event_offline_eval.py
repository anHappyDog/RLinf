#!/usr/bin/env python3
"""Evaluate whether a short-memory policy generates demonstrated grasp events."""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections import defaultdict, deque
from pathlib import Path

import torch
from openpi.models.tokenizer import PaligemmaTokenizer

from rlinf.data.datasets.openpi_rlinf.behavior.behavior_sft_data_loader import (
    BehaviorSftDataLoader,
    create_behavior_sft_data_loader,
)
from rlinf.models.embodiment.openpi_rlinf.pi0_model.model import Observation
from toolkits.mem.grasp_event_types import GraspEventSample
from toolkits.mem.short_memory_offline_eval import (
    _CONDITIONS,
    ablate_observation_history,
    load_short_memory_model,
    move_observation_to_device,
)

_PHASES = ("open_control", "close_onset", "closed_hold")
_ACTION_GROUPS = {
    "base": slice(0, 3),
    "trunk": slice(3, 7),
    "left_arm": slice(7, 14),
    "left_gripper": slice(14, 15),
    "right_arm": slice(15, 22),
    "right_gripper": slice(22, 23),
}


def classify_gripper_phase(
    actions: torch.Tensor,
    gripper_indices: tuple[int, int] = (14, 22),
    close_threshold: float = 0.0,
) -> str:
    """Classify an action chunk as open, close-onset, closed-hold, or other."""
    grippers = actions[:, list(gripper_indices)]
    any_closed = (grippers < close_threshold).any(dim=-1)
    if not bool(any_closed.any()):
        return "open"
    if not bool(any_closed[0]):
        return "close_onset"
    if float(any_closed.float().mean()) >= 0.75:
        return "closed_hold"
    return "other"


def first_close_index(
    actions: torch.Tensor,
    gripper_indices: tuple[int, int] = (14, 22),
    close_threshold: float = 0.0,
) -> int | None:
    """Return the first timestep where either demonstrated gripper is closed."""
    grippers = actions[:, list(gripper_indices)]
    indices = torch.nonzero(
        (grippers < close_threshold).any(dim=-1), as_tuple=False
    ).flatten()
    return int(indices[0]) if len(indices) else None


class GraspEventSelector:
    """Select diverse, near-event pickup samples from a sequential stream."""

    def __init__(
        self,
        *,
        samples_per_phase: int,
        max_samples_per_episode: int,
        open_control_margin: int,
        closed_sample_stride: int,
        gripper_indices: tuple[int, int],
        close_threshold: float,
    ) -> None:
        self.samples_per_phase = samples_per_phase
        self.max_samples_per_episode = max_samples_per_episode
        self.open_control_margin = open_control_margin
        self.closed_sample_stride = closed_sample_stride
        self.gripper_indices = gripper_indices
        self.close_threshold = close_threshold
        self.selected = {phase: [] for phase in _PHASES}
        self._recent_open: dict[int, deque[GraspEventSample]] = defaultdict(
            lambda: deque(maxlen=open_control_margin + max_samples_per_episode)
        )
        self._controls_committed: set[int] = set()
        self._onset_buckets: dict[int, set[int]] = defaultdict(set)
        self._closed_seen: dict[int, int] = defaultdict(int)
        self._phase_episode_counts: dict[str, dict[int, int]] = {
            phase: defaultdict(int) for phase in _PHASES
        }

    @property
    def done(self) -> bool:
        """Return whether every phase has reached its requested sample count."""
        return all(
            len(self.selected[phase]) >= self.samples_per_phase for phase in _PHASES
        )

    def _append(self, sample: GraspEventSample) -> None:
        phase_samples = self.selected[sample.phase]
        episode_count = self._phase_episode_counts[sample.phase][sample.episode_index]
        if (
            len(phase_samples) >= self.samples_per_phase
            or episode_count >= self.max_samples_per_episode
        ):
            return
        phase_samples.append(sample)
        self._phase_episode_counts[sample.phase][sample.episode_index] += 1

    def consider(
        self,
        *,
        episode_index: int,
        frame_index: int,
        observation: Observation,
        actions: torch.Tensor,
    ) -> None:
        """Consume one pickup sample and retain it when it fills an event slot."""
        phase = classify_gripper_phase(
            actions, self.gripper_indices, self.close_threshold
        )
        sample = GraspEventSample(
            phase=phase,
            episode_index=episode_index,
            frame_index=frame_index,
            valid_history_frames=int(observation.history_frame_mask.sum()),
            observation=observation,
            actions=actions.clone(),
        )
        if phase == "open":
            self._recent_open[episode_index].append(sample)
            return

        if phase == "close_onset":
            if episode_index not in self._controls_committed:
                controls = list(self._recent_open[episode_index])[
                    : self.max_samples_per_episode
                ]
                for control in controls:
                    self._append(dataclasses.replace(control, phase="open_control"))
                self._controls_committed.add(episode_index)

            onset_index = first_close_index(
                actions, self.gripper_indices, self.close_threshold
            )
            assert onset_index is not None
            bucket = min(
                self.max_samples_per_episode - 1,
                onset_index * self.max_samples_per_episode // actions.shape[0],
            )
            if bucket not in self._onset_buckets[episode_index]:
                self._append(sample)
                self._onset_buckets[episode_index].add(bucket)
            return

        if phase == "closed_hold":
            seen = self._closed_seen[episode_index]
            if seen % self.closed_sample_stride == 0:
                self._append(sample)
            self._closed_seen[episode_index] += 1

    def samples(self) -> list[GraspEventSample]:
        """Return selected samples grouped in stable phase order."""
        if not self.done:
            counts = {phase: len(self.selected[phase]) for phase in _PHASES}
            raise ValueError(f"Event selection is incomplete: {counts}.")
        return [sample for phase in _PHASES for sample in self.selected[phase]]


def repeat_observation(observation: Observation, repeats: int) -> Observation:
    """Repeat a batch-size-one observation for paired flow-noise draws."""

    def _repeat(value: torch.Tensor | None) -> torch.Tensor | None:
        if value is None:
            return None
        return value.repeat((repeats, *((1,) * (value.ndim - 1))))

    return Observation(
        images={key: _repeat(value) for key, value in observation.images.items()},
        image_masks={
            key: _repeat(value) for key, value in observation.image_masks.items()
        },
        state=_repeat(observation.state),
        tokenized_prompt=_repeat(observation.tokenized_prompt),
        tokenized_prompt_mask=_repeat(observation.tokenized_prompt_mask),
        token_ar_mask=_repeat(observation.token_ar_mask),
        token_loss_mask=_repeat(observation.token_loss_mask),
        pcd_xyz=_repeat(observation.pcd_xyz),
        history_states=_repeat(observation.history_states),
        history_frame_mask=_repeat(observation.history_frame_mask),
        history_time_offsets=_repeat(observation.history_time_offsets),
    )


def summarize_sample_prediction(
    *,
    sample: GraspEventSample,
    predicted_actions: torch.Tensor,
    gripper_indices: tuple[int, int] = (14, 22),
    close_threshold: float = 0.0,
) -> dict:
    """Measure generated gripper behavior for one sample and several draws."""
    ground_truth = sample.actions[:, list(gripper_indices)] < close_threshold
    predicted = predicted_actions[:, :, list(gripper_indices)] < close_threshold
    gt_any = ground_truth.any(dim=-1)
    pred_any = predicted.any(dim=-1)
    gt_hands = ground_truth.any(dim=0)
    pred_hands = predicted.any(dim=1)

    if bool(gt_hands.any()):
        correct_hand_rate = float(pred_hands[:, gt_hands].all(dim=-1).float().mean())
    else:
        correct_hand_rate = None
    if bool(gt_any.any()):
        closed_step_recall = float(pred_any[:, gt_any].float().mean())
        gt_first = int(torch.nonzero(gt_any, as_tuple=False)[0])
        pred_first = []
        for draw in pred_any:
            indices = torch.nonzero(draw, as_tuple=False).flatten()
            pred_first.append(
                int(indices[0]) if len(indices) else int(sample.actions.shape[0])
            )
        first_close_mae = sum(abs(index - gt_first) for index in pred_first) / len(
            pred_first
        )
    else:
        closed_step_recall = None
        gt_first = None
        first_close_mae = None

    ground_truth_actions = sample.actions.float()
    predicted_actions = predicted_actions.float()
    if predicted_actions.shape[1:] != ground_truth_actions.shape:
        raise ValueError(
            "Predicted and ground-truth action shapes do not align: "
            f"{tuple(predicted_actions.shape[1:])} != "
            f"{tuple(ground_truth_actions.shape)}."
        )
    action_metrics = {
        f"action_mae_{name}": float(
            (
                predicted_actions[:, :, indices]
                - ground_truth_actions[None, :, indices]
            )
            .abs()
            .mean()
        )
        for name, indices in _ACTION_GROUPS.items()
    }
    non_gripper_indices = [
        index
        for index in range(23)
        if index not in set(gripper_indices)
    ]
    predicted_arm = predicted_actions[:, :, non_gripper_indices].flatten(1)
    target_arm = ground_truth_actions[:, non_gripper_indices].flatten()
    action_metrics["non_gripper_action_cosine"] = float(
        torch.nn.functional.cosine_similarity(
            predicted_arm,
            target_arm.unsqueeze(0).expand_as(predicted_arm),
            dim=-1,
        ).mean()
    )

    return {
        "phase": sample.phase,
        "episode_index": sample.episode_index,
        "frame_index": sample.frame_index,
        "valid_history_frames": sample.valid_history_frames,
        "gt_first_close_index": gt_first,
        "gt_closed_step_fraction": float(gt_any.float().mean()),
        "gt_left_closes": bool(gt_hands[0]),
        "gt_right_closes": bool(gt_hands[1]),
        "pred_any_close_draw_rate": float(pred_any.any(dim=-1).float().mean()),
        "pred_closed_step_fraction": float(pred_any.float().mean()),
        "pred_left_close_draw_rate": float(pred_hands[:, 0].float().mean()),
        "pred_right_close_draw_rate": float(pred_hands[:, 1].float().mean()),
        "correct_hand_close_draw_rate": correct_hand_rate,
        "closed_step_recall": closed_step_recall,
        "first_close_index_mae_with_miss_penalty": first_close_mae,
        **action_metrics,
    }


def summarize_records(
    records: list[dict],
    *,
    min_event_any_close_rate: float,
    min_correct_hand_close_rate: float,
    min_closed_step_recall: float,
    max_open_any_close_rate: float,
) -> dict:
    """Aggregate phase metrics and apply the explicit offline grasp gate."""

    def _mean(items: list[dict], key: str) -> float | None:
        values = [item.get(key) for item in items if item.get(key) is not None]
        return sum(values) / len(values) if values else None

    per_phase = {}
    for phase in _PHASES:
        items = [record for record in records if record["phase"] == phase]
        per_phase[phase] = {"num_samples": len(items)}
        for key in (
            "valid_history_frames",
            "gt_closed_step_fraction",
            "pred_any_close_draw_rate",
            "pred_closed_step_fraction",
            "pred_left_close_draw_rate",
            "pred_right_close_draw_rate",
            "correct_hand_close_draw_rate",
            "closed_step_recall",
            "first_close_index_mae_with_miss_penalty",
            *[f"action_mae_{name}" for name in _ACTION_GROUPS],
            "non_gripper_action_cosine",
        ):
            per_phase[phase][key] = _mean(items, key)

    event_records = [record for record in records if record["phase"] != "open_control"]
    event_any_close_rate = _mean(event_records, "pred_any_close_draw_rate")
    correct_hand_close_rate = _mean(event_records, "correct_hand_close_draw_rate")
    closed_step_recall = _mean(event_records, "closed_step_recall")
    open_any_close_rate = per_phase["open_control"]["pred_any_close_draw_rate"]
    gate = (
        event_any_close_rate is not None
        and event_any_close_rate >= min_event_any_close_rate
        and correct_hand_close_rate is not None
        and correct_hand_close_rate >= min_correct_hand_close_rate
        and closed_step_recall is not None
        and closed_step_recall >= min_closed_step_recall
        and open_any_close_rate is not None
        and open_any_close_rate <= max_open_any_close_rate
    )
    return {
        "per_phase": per_phase,
        "gate_metrics": {
            "event_any_close_rate": event_any_close_rate,
            "correct_hand_close_rate": correct_hand_close_rate,
            "closed_step_recall": closed_step_recall,
            "open_any_close_rate": open_any_close_rate,
        },
        "gate_thresholds": {
            "min_event_any_close_rate": min_event_any_close_rate,
            "min_correct_hand_close_rate": min_correct_hand_close_rate,
            "min_closed_step_recall": min_closed_step_recall,
            "max_open_any_close_rate": max_open_any_close_rate,
        },
        "offline_grasp_gate": gate,
        "action_fidelity": {
            **{
                f"action_mae_{name}": _mean(records, f"action_mae_{name}")
                for name in _ACTION_GROUPS
            },
            "non_gripper_action_cosine": _mean(
                records, "non_gripper_action_cosine"
            ),
        },
        "per_sample": records,
    }


def summarize_history_comparison(records_by_condition: dict[str, list[dict]]) -> dict:
    """Aggregate paired action-fidelity deltas against correct history."""
    if "correct" not in records_by_condition:
        raise ValueError("History comparison requires the correct condition.")
    correct = records_by_condition["correct"]
    metrics = [
        *[f"action_mae_{name}" for name in _ACTION_GROUPS],
        "non_gripper_action_cosine",
    ]
    result = {}
    for condition, records in records_by_condition.items():
        if condition == "correct":
            continue
        if len(records) != len(correct):
            raise ValueError("Paired history conditions have different sample counts.")
        condition_result = {}
        for metric in metrics:
            deltas = [
                controlled[metric] - baseline[metric]
                for baseline, controlled in zip(correct, records, strict=True)
            ]
            lower_is_better = metric.startswith("action_mae_")
            wins = [
                baseline[metric] < controlled[metric]
                if lower_is_better
                else baseline[metric] > controlled[metric]
                for baseline, controlled in zip(correct, records, strict=True)
            ]
            condition_result[metric] = {
                "mean_delta_controlled_minus_correct": sum(deltas) / len(deltas),
                "correct_win_rate": sum(wins) / len(wins),
            }
        result[condition] = condition_result
    return result


def assemble_history_metrics(
    condition_metrics: dict[str, dict], records_by_condition: dict[str, list[dict]]
) -> dict:
    """Attach paired condition summaries without creating self-references."""
    if "correct" not in condition_metrics:
        raise ValueError("History metrics require the correct condition.")
    metrics = dict(condition_metrics["correct"])
    metrics["history_conditions"] = condition_metrics
    metrics["paired_history_comparison"] = summarize_history_comparison(
        records_by_condition
    )
    return metrics


def _stream_position(loader: BehaviorSftDataLoader) -> tuple[int, int]:
    """Read the position of the item just yielded by a single-process loader."""
    transformed_dataset = loader.torch_loader.dataset
    dataset = transformed_dataset._dataset  # noqa: SLF001
    item = dataset.hf_dataset[dataset.current_streaming_frame_idx - 1]
    episode_index = int(item["episode_index"])
    frame_index = round(float(item["timestamp"]) * dataset.fps)
    return episode_index, frame_index


def _is_pickup_prompt(
    observation: Observation,
    pickup_tokens: torch.Tensor,
    pickup_mask: torch.Tensor,
) -> bool:
    return bool(
        torch.equal(observation.tokenized_prompt[0], pickup_tokens)
        and torch.equal(observation.tokenized_prompt_mask[0], pickup_mask)
    )


def _create_loader(args) -> BehaviorSftDataLoader:
    return create_behavior_sft_data_loader(
        behavior_dataset_root=args.dataset_root,
        assets_dir=args.assets_dir,
        asset_id=args.asset_id,
        model_path=args.checkpoint,
        config_name="pi05_behavior",
        repo_id="behavior-1k/2025-challenge-demos",
        tasks=[args.task],
        modalities=["rgb"],
        action_dim=32,
        action_horizon=32,
        max_token_len=200,
        batch_size=1,
        num_workers=0,
        fine_grained_level=0,
        tolerance_s=1.0e-4,
        shuffle=False,
        seed=args.seed,
        skill_labels=None,
        use_skill=False,
        prompt_source="primitive",
        primitive_prompt_probability=1.0,
        mixed_boundary_fallback_to_task=False,
        history_length=6,
        history_frame_stride=30,
        history_state_dim=23,
        discrete_state_input=False,
        enable_gap=True,
        allow_left=0,
        allow_right=0,
        dist_rank=0,
        dist_world_size=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--asset-id", default="physical-intelligence/behavior")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--selection-cache",
        help="Optional path to a reusable torch cache of selected event samples.",
    )
    parser.add_argument(
        "--reuse-selection-cache",
        action="store_true",
        help="Load selected event samples instead of scanning the dataset.",
    )
    parser.add_argument("--task", default="turning_on_radio")
    parser.add_argument("--pickup-prompt", default="pick up radio from coffee table")
    parser.add_argument("--samples-per-phase", type=int, default=16)
    parser.add_argument("--max-samples-per-episode", type=int, default=2)
    parser.add_argument(
        "--open-control-margin",
        type=int,
        default=32,
        help="Minimum streamed frames between open controls and onset candidates.",
    )
    parser.add_argument("--closed-sample-stride", type=int, default=24)
    parser.add_argument("--max-scanned-samples", type=int, default=50_000)
    parser.add_argument("--noise-draws", type=int, default=4)
    parser.add_argument("--denoise-steps", type=int, default=5)
    parser.add_argument(
        "--history-conditions",
        nargs="+",
        choices=_CONDITIONS,
        default=("correct",),
        help="Paired history controls evaluated with identical flow noise.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gripper-indices", type=int, nargs=2, default=(14, 22))
    parser.add_argument("--close-threshold", type=float, default=0.0)
    parser.add_argument("--min-event-any-close-rate", type=float, default=0.5)
    parser.add_argument("--min-correct-hand-close-rate", type=float, default=0.5)
    parser.add_argument("--min-closed-step-recall", type=float, default=0.5)
    parser.add_argument("--max-open-any-close-rate", type=float, default=0.25)
    args = parser.parse_args()

    if args.samples_per_phase <= 0 or args.max_samples_per_episode <= 0:
        raise ValueError("Sample counts must be positive.")
    if args.open_control_margin < 0:
        raise ValueError("Open-control margin must be non-negative.")
    if args.noise_draws <= 0 or args.denoise_steps <= 0:
        raise ValueError("Noise draws and denoise steps must be positive.")
    history_conditions = tuple(dict.fromkeys(args.history_conditions))
    if "correct" not in history_conditions:
        raise ValueError("--history-conditions must include correct.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_cache = (
        Path(args.selection_cache).expanduser().resolve()
        if args.selection_cache
        else output_dir / "selection_samples.pt"
    )
    gripper_indices = tuple(args.gripper_indices)
    if args.reuse_selection_cache:
        if not selection_cache.is_file():
            raise FileNotFoundError(f"Selection cache not found: {selection_cache}")
        samples = torch.load(selection_cache, map_location="cpu", weights_only=False)
        scanned = 0
        print(f"Loaded {len(samples)} event samples from {selection_cache}.")
    else:
        loader = _create_loader(args)
        pickup_tokenizer = PaligemmaTokenizer(max_len=200)
        pickup_tokens, pickup_mask = pickup_tokenizer.tokenize(args.pickup_prompt, None)
        pickup_tokens = torch.from_numpy(pickup_tokens).long()
        pickup_mask = torch.from_numpy(pickup_mask).bool()
        selector = GraspEventSelector(
            samples_per_phase=args.samples_per_phase,
            max_samples_per_episode=args.max_samples_per_episode,
            open_control_margin=args.open_control_margin,
            closed_sample_stride=args.closed_sample_stride,
            gripper_indices=gripper_indices,
            close_threshold=args.close_threshold,
        )

        scanned = 0
        previous_episode = None
        for observation, actions in loader:
            scanned += 1
            if scanned > args.max_scanned_samples:
                break
            if not _is_pickup_prompt(observation, pickup_tokens, pickup_mask):
                continue
            episode_index, frame_index = _stream_position(loader)
            if previous_episode is not None and episode_index != previous_episode:
                counts = {phase: len(selector.selected[phase]) for phase in _PHASES}
                print(f"Finished episode {previous_episode}: {counts}", flush=True)
            previous_episode = episode_index
            selector.consider(
                episode_index=episode_index,
                frame_index=frame_index,
                observation=observation,
                actions=actions[0],
            )
            if selector.done:
                break
        samples = selector.samples()
        selection_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(samples, selection_cache)
        counts = {
            phase: sum(sample.phase == phase for sample in samples) for phase in _PHASES
        }
        print(
            "Selected "
            + ", ".join(f"{phase}={counts[phase]}" for phase in _PHASES)
            + f" after scanning {scanned} samples. Cached at {selection_cache}."
        )
    selection_manifest = [
        {
            "phase": sample.phase,
            "episode_index": sample.episode_index,
            "frame_index": sample.frame_index,
            "valid_history_frames": sample.valid_history_frames,
            "gt_first_close_index": first_close_index(
                sample.actions, gripper_indices, args.close_threshold
            ),
        }
        for sample in samples
    ]
    selection_path = output_dir / "selection_manifest.json"
    selection_path.write_text(
        json.dumps(selection_manifest, indent=2) + "\n", encoding="utf-8"
    )
    device = torch.device("cuda")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    model, checkpoint_source = load_short_memory_model(checkpoint, device)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    records_by_condition = {condition: [] for condition in history_conditions}
    with torch.no_grad():
        for index, sample in enumerate(samples, start=1):
            observation = move_observation_to_device(sample.observation, device)
            noise = torch.randn(
                args.noise_draws,
                sample.actions.shape[0],
                sample.actions.shape[1],
                device=device,
                dtype=torch.bfloat16,
                generator=generator,
            )
            for condition in history_conditions:
                controlled = ablate_observation_history(observation, condition)
                controlled = repeat_observation(controlled, args.noise_draws)
                predicted_actions = model.sample_actions(
                    controlled,
                    num_steps=args.denoise_steps,
                    noise=noise,
                )
                records_by_condition[condition].append(
                    summarize_sample_prediction(
                        sample=sample,
                        predicted_actions=predicted_actions.float().cpu(),
                        gripper_indices=gripper_indices,
                        close_threshold=args.close_threshold,
                    )
                )
            print(f"Evaluated {index}/{len(samples)}", end="\r", flush=True)
    print()

    condition_metrics = {
        condition: summarize_records(
            records,
            min_event_any_close_rate=args.min_event_any_close_rate,
            min_correct_hand_close_rate=args.min_correct_hand_close_rate,
            min_closed_step_recall=args.min_closed_step_recall,
            max_open_any_close_rate=args.max_open_any_close_rate,
        )
        for condition, records in records_by_condition.items()
    }
    metrics = assemble_history_metrics(
        condition_metrics,
        records_by_condition,
    )
    metrics.update(
        checkpoint=str(checkpoint_source),
        task=args.task,
        pickup_prompt=args.pickup_prompt,
        seed=args.seed,
        noise_draws=args.noise_draws,
        denoise_steps=args.denoise_steps,
        evaluated_history_conditions=history_conditions,
        gripper_indices=gripper_indices,
        close_threshold=args.close_threshold,
        scanned_samples=scanned,
        selection_cache=str(selection_cache),
    )
    output_path = output_dir / "metrics.json"
    output_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in metrics.items() if key != "per_sample"},
            indent=2,
        )
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
