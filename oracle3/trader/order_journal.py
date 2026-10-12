"""Durable intent reservations and cumulative accounting checkpoints.

The journal prevents resubmission of a client key, including after a crash. It
is not an account inventory service: recovered sessions must verify inventory
before applying further fills. None selects an in-memory database for tests.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class OrderJournal:
    def __init__(self, path: str | Path | None = None, namespace: str = 'default'):
        if not isinstance(namespace, str) or not namespace.strip():
            raise ValueError('journal namespace must be nonempty')
        self.namespace = namespace
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(
            str(path) if path is not None else ':memory:', timeout=5
        )
        self._db.row_factory = sqlite3.Row
        self._db.execute('PRAGMA synchronous=FULL')
        self._db.execute("""CREATE TABLE IF NOT EXISTS orders (
            namespace TEXT NOT NULL, client_order_id TEXT NOT NULL,
            request TEXT NOT NULL, venue_order_id TEXT NOT NULL DEFAULT '',
            result TEXT, needs_reconciliation INTEGER NOT NULL DEFAULT 1,
            applied_quantity TEXT NOT NULL DEFAULT '0',
            applied_notional TEXT NOT NULL DEFAULT '0',
            applied_fee TEXT NOT NULL DEFAULT '0',
            PRIMARY KEY(namespace, client_order_id))""")
        self._db.commit()

    def reserve(self, client_order_id: str, request: dict[str, Any]) -> bool:
        if not isinstance(client_order_id, str) or not client_order_id.strip():
            raise ValueError('client_order_id must be nonempty')
        payload = json.dumps(request, allow_nan=False, sort_keys=True)
        with self._db:
            cursor = self._db.execute(
                'INSERT OR IGNORE INTO orders(namespace,client_order_id,request) VALUES(?,?,?)',
                (self.namespace, client_order_id, payload),
            )
        return cursor.rowcount == 1

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item['request'] = json.loads(item['request'])
        item['result'] = json.loads(item['result']) if item['result'] else None
        item['needs_reconciliation'] = bool(item['needs_reconciliation'])
        return item

    def get(self, client_order_id: str) -> dict[str, Any] | None:
        row = self._db.execute(
            'SELECT * FROM orders WHERE namespace=? AND client_order_id=?',
            (self.namespace, client_order_id),
        ).fetchone()
        return self._decode(row) if row else None

    def records(self, *, pending_only: bool = False) -> list[dict[str, Any]]:
        query = 'SELECT * FROM orders WHERE namespace=?'
        if pending_only:
            query += ' AND needs_reconciliation=1'
        return [self._decode(row) for row in self._db.execute(query, (self.namespace,))]

    def update(
        self,
        client_order_id: str,
        *,
        venue_order_id: str | None = None,
        result: dict[str, Any] | None = None,
        needs_reconciliation: bool = True,
        applied_quantity: str | None = None,
        applied_notional: str | None = None,
        applied_fee: str | None = None,
    ) -> None:
        changes: dict[str, Any] = {'needs_reconciliation': int(needs_reconciliation)}
        for key, value in [
            ('venue_order_id', venue_order_id),
            ('applied_quantity', applied_quantity),
            ('applied_notional', applied_notional),
            ('applied_fee', applied_fee),
        ]:
            if value is not None:
                changes[key] = value
        if result is not None:
            changes['result'] = json.dumps(result, allow_nan=False, sort_keys=True)
        assignment = ','.join(f'{key}=?' for key in changes)
        with self._db:
            cursor = self._db.execute(
                f'UPDATE orders SET {assignment} WHERE namespace=? AND client_order_id=?',
                (*changes.values(), self.namespace, client_order_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(client_order_id)

    def close(self) -> None:
        self._db.close()
