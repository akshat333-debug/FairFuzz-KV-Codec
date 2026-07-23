from pydantic import BaseModel

from fairfuzzkv_codec.unicode_grouping.schema import NormalizationPolicy


class UnicodeGroupingConfig(BaseModel):
    normalization_policy: NormalizationPolicy = NormalizationPolicy.PRESERVE_ORIGINAL
