// Copyright (c) 2026 1abcdefggs
// SPDX-License-Identifier: MIT
// https://github.com/1abcdefggs/vect-or-engine

use crate::profile::Profile;
use aho_corasick::AhoCorasick;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationMatch {
    pub item_id: String,
    pub item_name: String,
    pub detected_colloquial: String,
    pub templates: HashMap<String, String>, // Level "A", "B", "C", "D" -> text
}

pub struct Translator {
    ac: Option<AhoCorasick>,
    match_metadata: Vec<(String, String, String, HashMap<String, String>)>, // (item_id, item_name, colloquial_phrase, templates)
}

impl Translator {
    pub fn new(profile: &Profile) -> Self {
        let mut patterns = Vec::new();
        let mut match_metadata = Vec::new();

        if let Some(trans_rules) = &profile.conversational_translation_rules {
            for rule in &trans_rules.rules {
                for pattern in &rule.colloquial_patterns {
                    patterns.push(pattern.clone());
                    match_metadata.push((
                        rule.item_id.clone(),
                        rule.item_name.clone(),
                        pattern.clone(),
                        rule.templates.clone(),
                    ));
                }
            }
        }

        let ac = if !patterns.is_empty() {
            AhoCorasick::builder().build(&patterns).ok()
        } else {
            None
        };

        Translator { ac, match_metadata }
    }

    pub fn match_colloquial(&self, text: &str) -> Vec<TranslationMatch> {
        let mut results = Vec::new();

        if let Some(ac) = &self.ac {
            for mat in ac.find_iter(text) {
                let idx = mat.pattern().as_usize();
                if idx < self.match_metadata.len() {
                    let (item_id, item_name, detected, templates) = &self.match_metadata[idx];
                    results.push(TranslationMatch {
                        item_id: item_id.clone(),
                        item_name: item_name.clone(),
                        detected_colloquial: detected.clone(),
                        templates: templates.clone(),
                    });
                }
            }
        }

        results
    }
}
