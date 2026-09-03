# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Structured grounded-control annotations for BEHAVIOR-1K."""

from .annotation_parser import (
    AnnotationParseResult,
    ParsedSkillSegment,
    ParseIssue,
    ParseStatus,
    canonicalize_entity_name,
    parse_skill_annotation,
)
from .entity_resolver import (
    EntityResolver,
    object_name_from_prim_path,
    parse_instance_id_mapping,
)
from .grounding import (
    DEFAULT_CAMERA_PRIORITY,
    GroundingConfig,
    filter_mask_components,
    ground_button_instance_ids,
    ground_control_spec,
    ground_entity_argument,
    ground_instance_ids,
    grounding_from_mask,
    infer_functional_parts,
    mask_for_instance_ids,
    select_primary_grounding,
)
from .runtime import (
    GROUND_CONTROL_JSON_KEY,
    GroundedPromptController,
    SidecarControlIndex,
    episode_index_from_annotation_dir,
)
from .schema import (
    CameraID,
    EntityArgument,
    GroundedControlSpec,
    Grounding2D,
    PartArgument,
    Role,
)
from .serializer import (
    ControlProfile,
    ControlSerializer,
    SerializerOptions,
    bbox_location_tokens,
    location_token,
    quantize_coordinate,
)
from .skill_registry import (
    DEFAULT_SKILL_SIGNATURE_REGISTRY,
    ArgumentLayout,
    SkillSignature,
    SkillSignatureRegistry,
)
from .tokens import (
    STRUCTURAL_TOKENS,
    ReservedTokenAllocator,
    ReservedTokenMapping,
    TokenBinding,
    TokenizerCapabilities,
)

__all__ = [
    "DEFAULT_SKILL_SIGNATURE_REGISTRY",
    "AnnotationParseResult",
    "ArgumentLayout",
    "CameraID",
    "ControlProfile",
    "ControlSerializer",
    "DEFAULT_CAMERA_PRIORITY",
    "EntityResolver",
    "EntityArgument",
    "GroundedControlSpec",
    "Grounding2D",
    "GroundingConfig",
    "GROUND_CONTROL_JSON_KEY",
    "GroundedPromptController",
    "SidecarControlIndex",
    "ParseIssue",
    "ParseStatus",
    "ParsedSkillSegment",
    "PartArgument",
    "Role",
    "STRUCTURAL_TOKENS",
    "SerializerOptions",
    "SkillSignature",
    "SkillSignatureRegistry",
    "ReservedTokenAllocator",
    "ReservedTokenMapping",
    "TokenBinding",
    "TokenizerCapabilities",
    "bbox_location_tokens",
    "canonicalize_entity_name",
    "episode_index_from_annotation_dir",
    "filter_mask_components",
    "ground_control_spec",
    "ground_button_instance_ids",
    "ground_entity_argument",
    "ground_instance_ids",
    "grounding_from_mask",
    "infer_functional_parts",
    "location_token",
    "mask_for_instance_ids",
    "object_name_from_prim_path",
    "parse_skill_annotation",
    "parse_instance_id_mapping",
    "quantize_coordinate",
    "select_primary_grounding",
]
