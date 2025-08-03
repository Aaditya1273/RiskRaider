import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Union, Any
import aiosqlite
import sqlite3


@dataclass
class Warning:
    """Data class representing a warning record."""
    id: int
    user_id: int
    server_id: int
    moderator_id: int
    reason: str
    created_at: datetime
    
    @classmethod
    def from_row(cls, row: tuple) -> 'Warning':
        """Create Warning instance from database row."""
        return cls(
            id=row[5],
            user_id=row[0],
            server_id=row[1],
            moderator_id=row[2],
            reason=row[3],
            created_at=datetime.fromtimestamp(int(row[4]))
        )


class DatabaseError(Exception):
    """Custom exception for database-related errors."""
    pass


class DatabaseManager:
    """
    Enhanced database manager with improved performance, error handling,
    and connection management capabilities.
    """
    
    def __init__(self, db_path: Union[str, Path], *, pool_size: int = 10,
                 timeout: float = 30.0, check_same_thread: bool = False) -> None:
        """
        Initialize the database manager.
        
        :param db_path: Path to the SQLite database file
        :param pool_size: Maximum number of connections in the pool
        :param timeout: Connection timeout in seconds
        :param check_same_thread: Whether to check same thread access
        """
        self.db_path = Path(db_path)
        self.pool_size = pool_size
        self.timeout = timeout
        self.check_same_thread = check_same_thread
        self._connection_pool: List[aiosqlite.Connection] = []
        self._pool_lock = asyncio.Lock()
        self._initialized = False
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self) -> None:
        """Initialize the database and create tables if they don't exist."""
        if self._initialized:
            return
            
        try:
            self._initialized = True
            # Ensure database directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create connection pool
            await self._create_connection_pool()
            
            # Initialize database schema
            await self._initialize_schema()
            
            self.logger.info(f"Database manager initialized with {self.pool_size} connections")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise DatabaseError(f"Database initialization failed: {e}")
    
    async def _create_connection_pool(self) -> None:
        """Create a pool of database connections."""
        for _ in range(self.pool_size):
            conn = await aiosqlite.connect(
                self.db_path,
                timeout=self.timeout,
                check_same_thread=self.check_same_thread
            )
            # Enable WAL mode for better concurrency
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA synchronous=NORMAL")
            await conn.execute("PRAGMA cache_size=10000")
            await conn.execute("PRAGMA temp_store=MEMORY")
            await conn.commit()
            self._connection_pool.append(conn)
    
    async def _initialize_schema(self) -> None:
        """Initialize database schema with optimized tables and indexes."""
        schema_sql = """
        CREATE TABLE IF NOT EXISTS warns (
            id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            server_id INTEGER NOT NULL,
            moderator_id INTEGER NOT NULL,
            reason TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT TRUE,
            severity INTEGER DEFAULT 1,
            PRIMARY KEY (id, user_id, server_id),
            FOREIGN KEY (user_id, server_id) REFERENCES users(id, server_id) ON DELETE CASCADE
        );
        
        CREATE INDEX IF NOT EXISTS idx_warns_user_server ON warns(user_id, server_id);
        CREATE INDEX IF NOT EXISTS idx_warns_moderator ON warns(moderator_id);
        CREATE INDEX IF NOT EXISTS idx_warns_created_at ON warns(created_at);
        CREATE INDEX IF NOT EXISTS idx_warns_active ON warns(is_active);
        
        CREATE TABLE IF NOT EXISTS warn_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            warn_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            server_id INTEGER NOT NULL,
            action TEXT NOT NULL,
            performed_by INTEGER NOT NULL,
            performed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            details TEXT
        );
        
        CREATE INDEX IF NOT EXISTS idx_warn_history_warn ON warn_history(warn_id, user_id, server_id);
        
        CREATE TRIGGER IF NOT EXISTS update_warns_timestamp 
        AFTER UPDATE ON warns
        BEGIN
            UPDATE warns SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id AND user_id = NEW.user_id AND server_id = NEW.server_id;
        END;
        """
        
        async with self._get_connection() as conn:
            await conn.executescript(schema_sql)
            await conn.commit()
    
    @asynccontextmanager
    async def _get_connection(self) -> AsyncGenerator[aiosqlite.Connection, None]:
        """Get a connection from the pool with automatic return."""
        if not self._initialized:
            # This check prevents using the manager before it's ready.
            raise DatabaseError("DatabaseManager is not initialized. Call initialize() first.")
            
        conn = None
        try:
            async with self._pool_lock:
                if not self._connection_pool:
                    # Wait for a connection to be returned to the pool
                    self.logger.warning("Connection pool empty. Waiting for a connection.")
                    # A more robust implementation might use an asyncio.Condition
                    # to wait here instead of raising an error immediately.
                    await asyncio.sleep(0.1) # Simple wait
                    if not self._connection_pool:
                        raise DatabaseError("No available connections in pool and timeout exceeded")
                conn = self._connection_pool.pop()
            
            yield conn
        finally:
            if conn:
                async with self._pool_lock:
                    self._connection_pool.append(conn)
    
    async def add_warn(self, user_id: int, server_id: int, moderator_id: int, 
                      reason: str, severity: int = 1) -> int:
        """
        Add a warning to the database with enhanced features.
        
        :param user_id: The ID of the user to be warned
        :param server_id: The ID of the server
        :param moderator_id: The ID of the moderator issuing the warning
        :param reason: The reason for the warning
        :param severity: Warning severity level (1-5)
        :return: The warn ID
        """
        if not (1 <= severity <= 5):
            raise ValueError("Severity must be between 1 and 5")
            
        async with self._get_connection() as conn:
            # Get next warn ID for this user/server combination
            async with conn.execute(
                "SELECT COALESCE(MAX(id), 0) + 1 FROM warns WHERE user_id=? AND server_id=?",
                (user_id, server_id)
            ) as cursor:
                result = await cursor.fetchone()
                warn_id = result[0]
            
            # Insert the warning
            await conn.execute(
                """INSERT INTO warns (id, user_id, server_id, moderator_id, reason, severity)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (warn_id, user_id, server_id, moderator_id, reason, severity)
            )
            
            # Log the action in history
            await conn.execute(
                """INSERT INTO warn_history (warn_id, user_id, server_id, action, performed_by, details)
                   VALUES (?, ?, ?, 'CREATED', ?, ?)""",
                (warn_id, user_id, server_id, moderator_id, f"Severity: {severity}")
            )
            
            await conn.commit()
            self.logger.info(f"Added warn {warn_id} for user {user_id} in server {server_id}")
            return warn_id
    
    async def remove_warn(self, warn_id: int, user_id: int, server_id: int, 
                         moderator_id: Optional[int] = None) -> int:
        """
        Remove a warning from the database (soft delete).
        
        :param warn_id: The ID of the warning
        :param user_id: The ID of the warned user
        :param server_id: The ID of the server
        :param moderator_id: The ID of the moderator removing the warning
        :return: Number of remaining active warnings
        """
        async with self._get_connection() as conn:
            # Soft delete the warning
            result = await conn.execute(
                "UPDATE warns SET is_active = FALSE WHERE id=? AND user_id=? AND server_id=? AND is_active=TRUE",
                (warn_id, user_id, server_id)
            )
            
            if result.rowcount == 0:
                raise DatabaseError(f"Warning {warn_id} not found or already inactive")
            
            # Log the action
            if moderator_id:
                await conn.execute(
                    """INSERT INTO warn_history (warn_id, user_id, server_id, action, performed_by)
                       VALUES (?, ?, ?, 'REMOVED', ?)""",
                    (warn_id, user_id, server_id, moderator_id)
                )
            
            # Count remaining active warnings
            async with conn.execute(
                "SELECT COUNT(*) FROM warns WHERE user_id=? AND server_id=? AND is_active=TRUE",
                (user_id, server_id)
            ) as cursor:
                result = await cursor.fetchone()
                remaining_warns = result[0] if result else 0
            
            await conn.commit()
            self.logger.info(f"Removed warn {warn_id} for user {user_id}, {remaining_warns} warnings remaining")
            return remaining_warns
    
    async def get_warnings(self, user_id: int, server_id: int, 
                          include_inactive: bool = False, limit: Optional[int] = None) -> List[Warning]:
        """
        Get all warnings for a user with enhanced filtering.
        
        :param user_id: The ID of the user
        :param server_id: The ID of the server
        :param include_inactive: Whether to include removed warnings
        :param limit: Maximum number of warnings to return
        :return: List of Warning objects
        """
        query = """
            SELECT user_id, server_id, moderator_id, reason, strftime('%s', created_at), id, severity
            FROM warns 
            WHERE user_id=? AND server_id=?
        """
        params = [user_id, server_id]
        
        if not include_inactive:
            query += " AND is_active=TRUE"
        
        query += " ORDER BY created_at DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        async with self._get_connection() as conn:
            async with conn.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [Warning.from_row(row) for row in rows]
    
    async def get_warning_stats(self, server_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Get comprehensive warning statistics.
        
        :param server_id: Optional server ID to filter by
        :return: Dictionary containing various statistics
        """
        base_query = "FROM warns WHERE is_active=TRUE"
        params = []
        
        if server_id:
            base_query += " AND server_id=?"
            params.append(server_id)
        
        async with self._get_connection() as conn:
            stats = {}
            
            # Total active warnings
            async with conn.execute(f"SELECT COUNT(*) {base_query}", params) as cursor:
                result = await cursor.fetchone()
                stats['total_active_warnings'] = result[0]
            
            # Warnings by severity
            async with conn.execute(f"SELECT severity, COUNT(*) {base_query} GROUP BY severity", params) as cursor:
                severity_stats = await cursor.fetchall()
                stats['by_severity'] = {row[0]: row[1] for row in severity_stats}
            
            # Most warned users
            async with conn.execute(
                f"SELECT user_id, COUNT(*) as warn_count {base_query} GROUP BY user_id ORDER BY warn_count DESC LIMIT 10",
                params
            ) as cursor:
                top_users = await cursor.fetchall()
                stats['most_warned_users'] = [{'user_id': row[0], 'count': row[1]} for row in top_users]
            
            # Most active moderators
            async with conn.execute(
                f"SELECT moderator_id, COUNT(*) as warn_count {base_query} GROUP BY moderator_id ORDER BY warn_count DESC LIMIT 10",
                params
            ) as cursor:
                top_mods = await cursor.fetchall()
                stats['most_active_moderators'] = [{'moderator_id': row[0], 'count': row[1]} for row in top_mods]
            
            return stats
    
    async def bulk_add_warns(self, warnings: List[Dict[str, Any]]) -> List[int]:
        """
        Add multiple warnings in a single transaction for better performance.
        
        :param warnings: List of warning dictionaries
        :return: List of generated warn IDs
        """
        warn_ids = []
        
        async with self._get_connection() as conn:
            for warning in warnings:
                # Get next warn ID
                async with conn.execute(
                    "SELECT COALESCE(MAX(id), 0) + 1 FROM warns WHERE user_id=? AND server_id=?",
                    (warning['user_id'], warning['server_id'])
                ) as cursor:
                    result = await cursor.fetchone()
                    warn_id = result[0]
                
                # Insert warning
                await conn.execute(
                    """INSERT INTO warns (id, user_id, server_id, moderator_id, reason, severity)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (warn_id, warning['user_id'], warning['server_id'], 
                     warning['moderator_id'], warning['reason'], warning.get('severity', 1))
                )
                
                warn_ids.append(warn_id)
            
            await conn.commit()
            self.logger.info(f"Bulk added {len(warnings)} warnings")
            return warn_ids
    
    async def search_warnings(self, query: str, server_id: Optional[int] = None,
                             limit: int = 50) -> List[Warning]:
        """
        Search warnings by reason text.
        
        :param query: Search query
        :param server_id: Optional server ID to filter by
        :param limit: Maximum number of results
        :return: List of matching warnings
        """
        sql = """
            SELECT user_id, server_id, moderator_id, reason, strftime('%s', created_at), id
            FROM warns 
            WHERE is_active=TRUE AND reason LIKE ?
        """
        params = [f"%{query}%"]
        
        if server_id:
            sql += " AND server_id=?"
            params.append(server_id)
        
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        async with self._get_connection() as conn:
            async with conn.execute(sql, params) as cursor:
                rows = await cursor.fetchall()
                return [Warning.from_row(row) for row in rows]
    
    async def cleanup_old_warnings(self, days_old: int = 365) -> int:
        """
        Archive old warnings to improve performance.
        
        :param days_old: Archive warnings older than this many days
        :return: Number of warnings archived
        """
        async with self._get_connection() as conn:
            result = await conn.execute(
                """UPDATE warns SET is_active = FALSE 
                   WHERE is_active = TRUE AND created_at < datetime('now', '-{} days')""".format(days_old)
            )
            archived_count = result.rowcount
            await conn.commit()
            
            self.logger.info(f"Archived {archived_count} old warnings")
            return archived_count
    
    async def close(self) -> None:
        """Close all connections in the pool."""
        async with self._pool_lock:
            for conn in self._connection_pool:
                await conn.close()
            self._connection_pool.clear()
        
        self._initialized = False
        self.logger.info("Database manager closed")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Factory function for easy instantiation
async def create_database_manager(db_path: Union[str, Path], **kwargs) -> DatabaseManager:
    """
    Factory function to create and initialize a DatabaseManager.
    
    :param db_path: Path to the database file
    :param kwargs: Additional arguments for DatabaseManager
    :return: Initialized DatabaseManager instance
    """
    manager = DatabaseManager(db_path, **kwargs)
    await manager.initialize()
    return manager


# Example usage and testing
async def main():
    """Example usage of the enhanced database manager."""
    async with DatabaseManager("warnings.db", pool_size=5) as db:
        # Add some warnings
        warn_id1 = await db.add_warn(12345, 67890, 11111, "Spam messages", severity=2)
        warn_id2 = await db.add_warn(12345, 67890, 11111, "Inappropriate language", severity=3)
        
        # Get warnings
        warnings = await db.get_warnings(12345, 67890)
        print(f"User has {len(warnings)} warnings")
        
        # Get statistics
        stats = await db.get_warning_stats(67890)
        print(f"Server statistics: {stats}")
        
        # Search warnings
        spam_warnings = await db.search_warnings("spam", 67890)
        print(f"Found {len(spam_warnings)} spam-related warnings")


if __name__ == "__main__":
    asyncio.run(main())