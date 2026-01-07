"""Version synchronization module for multi-instance cache consistency.

This module provides version synchronization mechanisms to ensure cache consistency
across multiple instances using ES polling strategy.
"""

import asyncio
import logging
from typing import Dict

from ..config import Config

logger = logging.getLogger(__name__)


class VersionSyncCoordinator:
    """Coordinates version synchronization across multiple instances.

    This class provides ES polling mechanism for maintaining cache consistency
    across multiple instances by periodically checking version updates in ES.

    Attributes:
        prompt_manager: The PromptManager instance to synchronize
        use_es_polling: Whether to use ES polling
        polling_task: Async task for ES polling (if enabled)
        polling_interval: Interval in seconds for ES polling
    """

    def __init__(self, prompt_manager, polling_interval: int = None):
        """Initialize the version sync coordinator.

        Args:
            prompt_manager: The PromptManager instance to synchronize
            polling_interval: Interval in seconds for ES polling (default: from config or 2)
        """
        self.prompt_manager = prompt_manager

        # Read polling interval from config, or use default value
        if polling_interval is None:
            polling_interval = Config.get_live_prompt_es_polling_interval()

        self.polling_interval = polling_interval
        self.use_es_polling = False
        self.polling_task = None
        self._is_running = False
        self._local_versions: Dict[str, int] = {}  # Track local versions
        self._pending_updates: Dict[str, set] = {}  # Track pending updates: {prompt_key: {version, ...}}

        # Detect ES polling availability
        self._detect_sync_mechanisms()

    def _detect_sync_mechanisms(self):
        """Detect ES polling availability based on configuration."""
        # Check ES configuration
        es_config = Config.get_es_config()
        if es_config and es_config.get("hosts"):
            es_hosts = es_config["hosts"]
            if isinstance(es_hosts, str):
                es_hosts = [es_hosts]

            # Check if any host is not local
            has_remote_es = any(
                host.split(":")[0] not in ["localhost", "127.0.0.1", "0.0.0.0"]
                for host in es_hosts
            )

            if has_remote_es:
                self.use_es_polling = True
                logger.info(f"ES polling enabled for remote hosts: {es_hosts}")
            else:
                logger.info("Local ES detected, polling disabled for multi-instance sync")
        else:
            logger.info("ES not configured, polling disabled")

        # If ES polling is not available, log a warning
        if not self.use_es_polling:
            logger.warning(
                "ES polling not available. "
                "Cache consistency across instances is not guaranteed."
            )

    async def start(self):
        """Start the version sync coordinator.

        Initializes ES polling based on configuration.
        """
        if self._is_running:
            logger.warning("Version sync coordinator is already running")
            return

        self._is_running = True

        # Initialize local versions from current cache
        self._initialize_local_versions()

        # Start ES polling if enabled
        if self.use_es_polling:
            await self._start_es_polling()

        logger.info("Version sync coordinator started")

    async def stop(self):
        """Stop the version sync coordinator.

        Cleans up ES polling tasks.
        """
        if not self._is_running:
            return

        self._is_running = False

        # Stop ES polling
        if self.use_es_polling and self.polling_task:
            await self._stop_es_polling()

        logger.info("Version sync coordinator stopped")

    def _initialize_local_versions(self):
        """Initialize local version tracking from current cache."""
        cache_snapshot = dict(self.prompt_manager._prompt_cache)
        for prompt_key, prompt_data in cache_snapshot.items():
            self._local_versions[prompt_key] = prompt_data.get("version", 1)

    async def _start_es_polling(self):
        """Start ES polling for version updates."""
        logger.info(
            f"Starting ES polling with {self.polling_interval}s interval "
            f"(configured in live_prompt.es_polling_interval)"
        )

        self.polling_task = asyncio.create_task(self._es_poller())

    async def _stop_es_polling(self):
        """Stop ES polling task."""
        if self.polling_task:
            self.polling_task.cancel()
            try:
                await self.polling_task
            except asyncio.CancelledError:
                pass
            logger.info("ES polling stopped")

    async def _es_poller(self):
        """Poll ES for version updates."""
        while self._is_running and self.use_es_polling:
            try:
                await self._check_es_versions()
                await asyncio.sleep(self.polling_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"ES polling error: {e}")
                await asyncio.sleep(self.polling_interval)

    async def _check_es_versions(self):
        """Check ES for version updates and update local cache if needed."""
        try:
            # Get all prompts from ES
            search_body = {
                "query": {"match_all": {}},
                "size": 1000,
                "_source": ["version", "updated_at"],
            }

            response = await self.prompt_manager.db_client.search(
                index_name=self.prompt_manager.index_name, body=search_body
            )

            if response is None:
                return

            hits = response.get("hits", {}).get("hits", [])

            for hit in hits:
                prompt_key = hit["_id"]
                remote_version = hit["_source"].get("version", 1)

                # Check if local version is behind
                local_version = self._local_versions.get(prompt_key, 0)

                if remote_version > local_version:
                    await self._handle_version_update(prompt_key, remote_version)

        except Exception as e:
            logger.error(f"Error checking ES versions: {e}")

    async def _handle_version_update(
        self, prompt_key: str, new_version: int
    ):
        """Handle a version update for a prompt with concurrency control.

        Prevents duplicate updates, version rollback, and out-of-order updates.

        Args:
            prompt_key: The prompt key to update
            new_version: The new version number
        """
        try:
            # Prevent duplicate updates for the same version
            if prompt_key in self._pending_updates and new_version in self._pending_updates[prompt_key]:
                logger.debug(f"Skipping duplicate update for {prompt_key} v{new_version}")
                return

            # Prevent version rollback: only accept if new_version > current version
            current_version = self._local_versions.get(prompt_key, 0)
            if new_version <= current_version:
                logger.debug(
                    f"Ignoring old version for {prompt_key}: new={new_version}, current={current_version}"
                )
                return

            # Mark as pending to prevent concurrent updates
            if prompt_key not in self._pending_updates:
                self._pending_updates[prompt_key] = set()
            self._pending_updates[prompt_key].add(new_version)

            try:
                # Fetch from ES with retry logic
                await self._fetch_from_es_with_retry(prompt_key, new_version)

            finally:
                # Remove from pending after processing (with cleanup of old versions)
                self._pending_updates[prompt_key].discard(new_version)
                # Clean up old versions to prevent memory leak (keep last 10)
                if len(self._pending_updates[prompt_key]) > 10:
                    # Remove oldest versions (keep higher version numbers)
                    versions_list = sorted(self._pending_updates[prompt_key])
                    versions_to_remove = versions_list[:len(versions_list) - 10]
                    for v in versions_to_remove:
                        self._pending_updates[prompt_key].discard(v)

        except Exception as e:
            logger.error(f"Failed to handle version update for {prompt_key}: {e}")

    async def _fetch_from_es_with_retry(self, prompt_key: str, new_version: int, max_retries: int = 3):
        """Fetch prompt from ES with retry logic to handle ES refresh delay.

        Args:
            prompt_key: The prompt key to fetch
            new_version: Expected version number
            max_retries: Maximum number of retries (default: 3)
        """
        import asyncio

        for attempt in range(max_retries):
            try:
                # Force fetch from ES (bypass cache)
                prompt_data = await self.prompt_manager.get_prompt(
                    prompt_key, use_cache=False
                )

                if prompt_data:
                    actual_version = prompt_data.get("version", 1)
                    if actual_version == new_version:
                        # Update local version tracker
                        self._local_versions[prompt_key] = new_version

                        logger.debug(
                            f"Cache updated for {prompt_key} from ES "
                            f"(version {new_version}, attempt {attempt + 1}/{max_retries})"
                        )

                        # Trigger hot-reload for agents using this prompt
                        from .wrapper import dynamic_agent_manager

                        await dynamic_agent_manager.update_prompt_by_key(prompt_key)
                        return
                    else:
                        logger.debug(
                            f"Version mismatch for {prompt_key} on attempt {attempt + 1}: "
                            f"expected={new_version}, got={actual_version}"
                        )
                        if actual_version > new_version:
                            # Newer version already exists, update our tracker and skip
                            self._local_versions[prompt_key] = actual_version
                            logger.info(f"Found newer version {actual_version} for {prompt_key}, skipping update")
                            return
                        # If actual_version < new_version, continue retrying
                else:
                    logger.debug(
                        f"Prompt {prompt_key} not found in ES during version sync "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )

            except Exception as e:
                logger.error(
                    f"Error fetching {prompt_key} from ES (attempt {attempt + 1}): {e}"
                )

            # Wait before retry (exponential backoff: 0.5s, 1s, 2s)
            if attempt < max_retries - 1:
                wait_time = 0.5 * (2 ** attempt)
                logger.debug(f"Retrying {prompt_key} after {wait_time}s...")
                await asyncio.sleep(wait_time)

        # All retries failed, log warning
        logger.warning(
            f"Failed to fetch {prompt_key} v{new_version} from ES after {max_retries} attempts. "
            f"Will sync on next ES polling cycle."
        )

    def update_local_version(self, prompt_key: str, version: int):
        """Update local version tracker (called after saving prompt).

        Args:
            prompt_key: The prompt key that was updated
            version: The new version number
        """
        self._local_versions[prompt_key] = version
        logger.debug(f"Updated local version tracker: {prompt_key} v{version}")


# Global version sync coordinator instance
version_sync_coordinator = None


async def get_version_sync_coordinator(prompt_manager=None) -> VersionSyncCoordinator:
    """Get the global version sync coordinator instance.

    Args:
        prompt_manager: The PromptManager instance to synchronize

    Returns:
        VersionSyncCoordinator: The global coordinator instance
    """
    global version_sync_coordinator

    if version_sync_coordinator is None:
        if prompt_manager is None:
            from .manager import get_prompt_manager

            prompt_manager = await get_prompt_manager()

        version_sync_coordinator = VersionSyncCoordinator(prompt_manager)

    return version_sync_coordinator


async def start_version_sync(prompt_manager=None):
    """Start the version sync coordinator.

    Args:
        prompt_manager: The PromptManager instance to synchronize
    """
    coordinator = await get_version_sync_coordinator(prompt_manager)
    await coordinator.start()


async def stop_version_sync():
    """Stop the version sync coordinator."""
    global version_sync_coordinator

    if version_sync_coordinator:
        await version_sync_coordinator.stop()
        version_sync_coordinator = None
