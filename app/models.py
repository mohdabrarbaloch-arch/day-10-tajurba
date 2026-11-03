"""ORM models.

Three tables, and the shape is the whole product:

  User      -> who owns the work
  Dataset   -> the uploaded CSV, stored as text so it can be re-parsed
  Run       -> one AutoML search: the leaderboard, the winner, the spec

A `Run` stores its leaderboard as JSON rather than as rows in a child table.
That is a deliberate trade: the leaderboard is written once and read whole,
never queried by candidate, so a child table would buy nothing and cost a
join on every page load.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(UTC)


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(
        String(255), unique=True, index=True, nullable=False
    )
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(20), default="analyst", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, server_default=func.now()
    )

    datasets: Mapped[list[Dataset]] = relationship(
        back_populates="owner", cascade="all, delete-orphan"
    )
    runs: Mapped[list[Run]] = relationship(
        back_populates="owner", cascade="all, delete-orphan"
    )


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    owner_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    # The raw CSV text. Storing it means a run can always be reproduced.
    content: Mapped[str] = mapped_column(Text, nullable=False)
    n_rows: Mapped[int] = mapped_column(Integer, default=0)
    n_columns: Mapped[int] = mapped_column(Integer, default=0)
    columns_json: Mapped[str] = mapped_column(Text, default="[]")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, server_default=func.now()
    )

    owner: Mapped[User] = relationship(back_populates="datasets")
    runs: Mapped[list[Run]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan"
    )


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    owner_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    dataset_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("datasets.id", ondelete="CASCADE"), index=True
    )

    target: Mapped[str] = mapped_column(String(200), nullable=False)
    task: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="completed", nullable=False)

    winner_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    winner_family: Mapped[str | None] = mapped_column(String(60), nullable=True)
    primary_metric: Mapped[str] = mapped_column(String(30), default="accuracy")
    primary_score: Mapped[float | None] = mapped_column(nullable=True)
    baseline_score: Mapped[float | None] = mapped_column(nullable=True)

    leaderboard_json: Mapped[str] = mapped_column(Text, default="[]")
    baseline_json: Mapped[str] = mapped_column(Text, default="{}")
    notes_json: Mapped[str] = mapped_column(Text, default="[]")
    spec_json: Mapped[str] = mapped_column(Text, default="{}")
    classes_json: Mapped[str] = mapped_column(Text, default="[]")

    n_rows: Mapped[int] = mapped_column(Integer, default=0)
    n_features: Mapped[int] = mapped_column(Integer, default=0)
    train_rows: Mapped[int] = mapped_column(Integer, default=0)
    test_rows: Mapped[int] = mapped_column(Integer, default=0)
    duration_seconds: Mapped[float] = mapped_column(default=0.0)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, server_default=func.now()
    )

    owner: Mapped[User] = relationship(back_populates="runs")
    dataset: Mapped[Dataset] = relationship(back_populates="runs")
