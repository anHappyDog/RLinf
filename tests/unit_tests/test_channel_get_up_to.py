import asyncio

import pytest

from rlinf.scheduler.channel.channel_worker import PeekQueue, WeightedItem, _get_up_to


def test_get_up_to_waits_for_first_then_collects_available_items():
    async def run():
        queue = PeekQueue()

        async def produce():
            await asyncio.sleep(0.01)
            await queue.put(WeightedItem(weight=0, item="first"))
            await queue.put(WeightedItem(weight=0, item="second"))

        producer = asyncio.create_task(produce())
        items = await _get_up_to(queue, max_items=3, timeout_seconds=0.02)
        await producer
        return items

    assert asyncio.run(run()) == ["first", "second"]


def test_get_up_to_stops_at_max_items_without_draining_queue():
    async def run():
        queue = PeekQueue()
        for item in range(3):
            await queue.put(WeightedItem(weight=0, item=item))
        items = await _get_up_to(queue, max_items=2, timeout_seconds=1.0)
        remaining = (await queue.get()).item
        return items, remaining

    assert asyncio.run(run()) == ([0, 1], 2)


@pytest.mark.parametrize(
    ("max_items", "timeout_seconds", "message"),
    [(0, 0.0, "max_items"), (1, -0.1, "timeout_seconds")],
)
def test_get_up_to_rejects_invalid_limits(max_items, timeout_seconds, message):
    async def run():
        await _get_up_to(
            PeekQueue(),
            max_items=max_items,
            timeout_seconds=timeout_seconds,
        )

    with pytest.raises(ValueError, match=message):
        asyncio.run(run())
