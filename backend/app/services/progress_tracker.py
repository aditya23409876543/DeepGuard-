"""
Real-time progress tracking for audio analysis.
Uses Server-Sent Events (SSE) to push updates to the frontend.
"""

import asyncio
import json
import logging
from typing import Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Manages progress updates for ongoing analysis tasks."""
    
    def __init__(self):
        self._queues: dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)
        self._completed: dict[str, dict] = {}
    
    async def create_task(self, task_id: str) -> None:
        """Create a new progress tracking task."""
        self._queues[task_id] = asyncio.Queue()
        self._completed[task_id] = None
        logger.info(f"Created progress task: {task_id}")
    
    async def update(self, task_id: str, stage: str, progress: float, message: str = "", details: dict = None) -> None:
        """Push a progress update for a task."""
        if task_id not in self._queues:
            await self.create_task(task_id)
        
        event = {
            "stage": stage,
            "progress": min(100, max(0, progress)),
            "message": message,
            "details": details or {}
        }
        
        await self._queues[task_id].put(event)
        logger.debug(f"Progress update for {task_id}: {stage} - {progress}%")
    
    async def complete(self, task_id: str, result: dict) -> None:
        """Mark a task as complete with final result."""
        if task_id in self._queues:
            await self._queues[task_id].put({
                "stage": "complete",
                "progress": 100,
                "message": "Analysis complete",
                "result": result
            })
        self._completed[task_id] = result
        logger.info(f"Task completed: {task_id}")
    
    async def error(self, task_id: str, error: str) -> None:
        """Mark a task as failed."""
        if task_id in self._queues:
            await self._queues[task_id].put({
                "stage": "error",
                "progress": 0,
                "message": error,
                "error": error
            })
    
    async def get_queue(self, task_id: str) -> asyncio.Queue:
        """Get the queue for a task (creates if not exists)."""
        if task_id not in self._queues:
            await self.create_task(task_id)
        return self._queues[task_id]
    
    def get_completed(self, task_id: str) -> Optional[dict]:
        """Get completed result if available."""
        return self._completed.get(task_id)
    
    async def cleanup(self, task_id: str) -> None:
        """Remove a task after completion."""
        if task_id in self._queues:
            del self._queues[task_id]
        if task_id in self._completed:
            del self._completed[task_id]


progress_tracker = ProgressTracker()
