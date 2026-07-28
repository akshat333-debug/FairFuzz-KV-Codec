"""Parallel per-language sentence templates.

Every language's `fact_owns` template renders "{name} owns code {digit}" in
that language/code-mix; every question template renders the same task
family's question. Simple, controlled fact statements - not idiomatic or
creative text - kept deliberately basic given these are LLM-authored and NOT
professionally reviewed (see package docstring). Grammatical imperfections
here do not affect the benchmark's PARALLELISM claim, which is verified
structurally (shared slot values, identical digit answer across languages),
not via translation-equivalence.
"""

from typing import Dict, NamedTuple

from fairfuzzkv_codec.benchmarks.indic_longcomp.schema import LanguageCondition


class LanguageTemplate(NamedTuple):
    fact_owns: str  # {name}, {digit}
    fact_hop_link: str  # {name1}, {name2} ("name1's teacher is name2")
    question_retrieval: str  # {name}
    question_multihop: str  # {name1}
    question_comparison: str  # {name1}, {name2}
    question_counting: str  # {digit}
    question_aggregation: str  # {name1}, {name2}


TEMPLATES: Dict[LanguageCondition, LanguageTemplate] = {
    LanguageCondition.ENGLISH: LanguageTemplate(
        fact_owns="Fact: {name} owns code {digit}.",
        fact_hop_link="Fact: {name1}'s teacher is {name2}.",
        question_retrieval=" Query: What is the code for {name}? Answer:",
        question_multihop=" Query: What is the code of {name1}'s teacher? Answer:",
        question_comparison=" Query: Between {name1} and {name2}, what is the larger of their codes? Answer:",
        question_counting=" Query: How many people own code {digit}? Answer:",
        question_aggregation=" Query: What is the sum of {name1}'s and {name2}'s codes? Answer:",
    ),
    LanguageCondition.HINDI: LanguageTemplate(
        fact_owns="तथ्य: {name} के पास कोड {digit} है।",
        fact_hop_link="तथ्य: {name1} के शिक्षक {name2} हैं।",
        question_retrieval=" प्रश्न: {name} का कोड क्या है? उत्तर:",
        question_multihop=" प्रश्न: {name1} के शिक्षक का कोड क्या है? उत्तर:",
        question_comparison=" प्रश्न: {name1} और {name2} में से जिसका कोड बड़ा है, वह कोड बताइए। उत्तर:",
        question_counting=" प्रश्न: कितने लोगों के पास कोड {digit} है? उत्तर:",
        question_aggregation=" प्रश्न: {name1} और {name2} के कोड का योग क्या है? उत्तर:",
    ),
    LanguageCondition.HINGLISH: LanguageTemplate(
        fact_owns="Fact: {name} ke paas code {digit} hai.",
        fact_hop_link="Fact: {name1} ke teacher {name2} hain.",
        question_retrieval=" Sawaal: {name} ka code kya hai? Jawab:",
        question_multihop=" Sawaal: {name1} ke teacher ka code kya hai? Jawab:",
        question_comparison=" Sawaal: {name1} aur {name2} mein se jiska code bada hai, wo code batao. Jawab:",
        question_counting=" Sawaal: kitne logon ke paas code {digit} hai? Jawab:",
        question_aggregation=" Sawaal: {name1} aur {name2} ke code ka jod kya hai? Jawab:",
    ),
    LanguageCondition.TELUGU_ENGLISH: LanguageTemplate(
        fact_owns="Fact: {name} daggara code {digit} undi.",
        fact_hop_link="Fact: {name1} teacher {name2}.",
        question_retrieval=" Prashna: {name} code enti? Samadhanam:",
        question_multihop=" Prashna: {name1} teacher code enti? Samadhanam:",
        question_comparison=" Prashna: {name1} mariyu {name2} lo evari code peddadi, aa code cheppu. Samadhanam:",
        question_counting=" Prashna: entha mandi daggara code {digit} undi? Samadhanam:",
        question_aggregation=" Prashna: {name1} mariyu {name2} codes yokka moththam enta? Samadhanam:",
    ),
}
