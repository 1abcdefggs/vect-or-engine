// Copyright (c) 2026 1abcdefggs
// SPDX-License-Identifier: MIT
// https://github.com/1abcdefggs/vect-or-engine

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Profile {
    pub metadata_standards: Option<MetadataStandards>,
    pub assessment_rules: Option<AssessmentRules>,
    pub conversational_translation_rules: Option<ConversationalTranslationRules>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MetadataStandards {
    pub authority: Option<String>,
    pub version: Option<String>,
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AssessmentRules {
    pub document_template: Option<DocumentTemplate>,
    pub keyword_rules: Option<Vec<KeywordRule>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct KeywordRule {
    pub trigger_words: Vec<String>,
    pub warning_message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DocumentTemplate {
    pub numerical_requirements: Option<NumericalRequirements>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NumericalRequirements {
    pub target_category: Option<String>,
    pub min_value: Option<u32>,
    pub value_regex_pattern: Option<String>,
    pub value_warning_template: Option<String>,
    pub conflict_items: Option<Vec<ConflictItem>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConflictItem {
    pub item_code: String,
    pub name: String,
    pub reason: String,
    pub max_allowed_value: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConversationalTranslationRules {
    pub description: Option<String>,
    pub rules: Vec<ConversationalRule>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConversationalRule {
    pub item_id: String,
    pub item_name: String,
    pub colloquial_patterns: Vec<String>,
    pub templates: HashMap<String, String>, // "A", "B", "C", "D" -> text
}

impl Profile {
    pub fn load_from_json_str(json_str: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json_str)
    }
}
