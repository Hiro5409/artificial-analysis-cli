"""Model families and the validated conditions for fetching a collection."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, assert_never

from artificial_analysis_cli._generated import PromptType as PromptType


def cli_name(name: str) -> str:
    """Spell a Python identifier the way the command line accepts it."""
    return name.lower().replace("_", "-")


class Family(Enum):
    """Models compared for the same use. Member names, spelled by `cli_name`, are the CLI aliases."""

    T2V = "text-to-video"
    I2V = "image-to-video"
    T2V_AUDIO = "text-to-video-audio"
    I2V_AUDIO = "image-to-video-audio"
    T2I = "text-to-image"
    IMAGE_EDITING = "image-editing"
    LLM = "language-models"
    TTS = "text-to-speech"
    STS = "speech-to-speech"
    STT = "speech-to-text"
    MUSIC_INSTRUMENTAL = "music-instrumental"
    MUSIC_VOCAL = "music-with-vocals"

    @property
    def alias(self) -> str:
        return cli_name(self.name)


class EndpointVariant(Enum):
    """The Free or Pro endpoint, which decides the scope and shape of the returned data."""

    FREE = "free"
    PRO = "pro"


AccountTier = Literal["free", "pro", "commercial"]
"""The tier of an API key's account. A Pro key can still read a Free endpoint variant."""

CategoryFamily = Literal[Family.T2V, Family.I2V, Family.T2V_AUDIO, Family.I2V_AUDIO, Family.T2I, Family.IMAGE_EDITING]
GenreFamily = Literal[Family.MUSIC_INSTRUMENTAL, Family.MUSIC_VOCAL]
SpeechFamily = Literal[Family.TTS, Family.STS, Family.STT]


@dataclass(frozen=True)
class FreeQuery:
    """Any family through its Free endpoint, which accepts no family-specific options."""

    family: Family
    variant: Literal[EndpointVariant.FREE] = field(default=EndpointVariant.FREE, init=False)


@dataclass(frozen=True)
class LanguageModelsProQuery:
    """Language models through the Pro endpoint; `None` leaves the prompt type to the API default."""

    prompt_type: PromptType | None = None
    family: Literal[Family.LLM] = field(default=Family.LLM, init=False)
    variant: Literal[EndpointVariant.PRO] = field(default=EndpointVariant.PRO, init=False)


@dataclass(frozen=True)
class CategoryProQuery:
    """An image or video family through the Pro endpoint, optionally with category breakdowns."""

    family: CategoryFamily
    include_categories: bool = False
    variant: Literal[EndpointVariant.PRO] = field(default=EndpointVariant.PRO, init=False)


@dataclass(frozen=True)
class GenreProQuery:
    """A music family through the Pro endpoint, optionally with genre breakdowns."""

    family: GenreFamily
    include_genres: bool = False
    variant: Literal[EndpointVariant.PRO] = field(default=EndpointVariant.PRO, init=False)


@dataclass(frozen=True)
class SpeechProQuery:
    """A speech family through the Pro endpoint, which accepts no family-specific options."""

    family: SpeechFamily
    variant: Literal[EndpointVariant.PRO] = field(default=EndpointVariant.PRO, init=False)


Query = FreeQuery | LanguageModelsProQuery | CategoryProQuery | GenreProQuery | SpeechProQuery


def options(query: Query) -> dict[str, str | bool]:
    """The family-specific API parameters the query sends; parameters left to their defaults are absent."""
    match query:
        case FreeQuery() | SpeechProQuery():
            return {}
        case LanguageModelsProQuery(prompt_type=prompt_type):
            return {} if prompt_type is None else {"prompt_type": prompt_type}
        case CategoryProQuery(include_categories=include_categories):
            return {"include_categories": True} if include_categories else {}
        case GenreProQuery(include_genres=include_genres):
            return {"include_genres": True} if include_genres else {}
        case _:
            assert_never(query)
