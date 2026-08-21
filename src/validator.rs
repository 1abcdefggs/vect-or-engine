// Copyright (c) 2026 1abcdefggs
// SPDX-License-Identifier: MIT
// https://github.com/1abcdefggs/vect-or-engine

use crate::profile::Profile;
use aho_corasick::AhoCorasick;
use regex::Regex;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationMarker {
    pub severity: String, // "Error" | "Warning" | "Info"
    pub message: String,
    pub line: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub extracted_value: Option<u32>,
    pub has_keyword_conflict: bool,
    pub keyword_message: Option<String>,
    pub value_message: Option<String>,
    /// SmallVec: zero heap allocation for ≤4 markers (typical case).
    pub markers: Vec<ValidationMarker>,
}

pub struct Validator {
    value_regex: Option<Regex>,
    value_warning_template: Option<String>,
    conflict_items_ac: Option<AhoCorasick>,
    conflict_item_reasons: Vec<String>,
    min_value: u32,
    keyword_rules: Vec<crate::profile::KeywordRule>,
}

impl Validator {
    pub fn new(profile: &Profile) -> Self {
        let mut patterns = Vec::new();
        let mut reasons = Vec::new();
        let mut min_value = 0;
        let mut value_regex = None;
        let mut value_warning_template = None;
        let mut keyword_rules = Vec::new();

        if let Some(rules) = &profile.assessment_rules {
            if let Some(template) = &rules.document_template {
                if let Some(reqs) = &template.numerical_requirements {
                    if let Some(min_v) = reqs.min_value {
                        min_value = min_v;
                    }
                    if let Some(pattern) = &reqs.value_regex_pattern {
                        if let Ok(re) = Regex::new(pattern) {
                            value_regex = Some(re);
                        }
                    }
                    if let Some(warning_template) = &reqs.value_warning_template {
                        value_warning_template = Some(warning_template.clone());
                    }
                    if let Some(conflicts) = &reqs.conflict_items {
                        for cd in conflicts {
                            patterns.push(cd.item_code.clone());
                            patterns.push(cd.name.clone());
                            reasons.push(format!("{}: {}", cd.name, cd.reason));
                            reasons.push(format!("{}: {}", cd.name, cd.reason));
                        }
                    }
                }
            }
            if let Some(keywords) = &rules.keyword_rules {
                keyword_rules = keywords.clone();
            }
        }

        let conflict_items_ac = if !patterns.is_empty() {
            AhoCorasick::builder().build(&patterns).ok()
        } else {
            None
        };

        Validator {
            value_regex,
            value_warning_template,
            conflict_items_ac,
            conflict_item_reasons: reasons,
            min_value,
            keyword_rules,
        }
    }

    pub fn validate(&self, text: &str) -> ValidationResult {
        // Pre-compute line start offsets once, reused for all markers.
        let line_starts: Vec<usize> = std::iter::once(0)
            .chain(text.char_indices().filter(|(_, c)| *c == '\n').map(|(i, _)| i + 1))
            .collect();

        // Map a byte offset to a 1-based line number.
        let offset_to_line = |offset: usize| -> u32 {
            line_starts.partition_point(|&s| s <= offset) as u32
        };

        let mut markers: Vec<ValidationMarker> = Vec::new();
        let mut extracted_value = None;
        let mut value_message = None;
        let mut has_keyword_conflict = false;
        let mut keyword_message = None;

        // 1. Extract numerical value
        if let Some(re) = &self.value_regex {
            if let Some(caps) = re.captures(text) {
                if let Some(m) = caps.get(1) {
                    if let Ok(val) = m.as_str().parse::<u32>() {
                        extracted_value = Some(val);
                        if val < self.min_value {
                            let msg = if let Some(template) = &self.value_warning_template {
                                template.replace("{min}", &self.min_value.to_string())
                                        .replace("{actual}", &val.to_string())
                            } else {
                                format!("Value requirement not met (Required: {}, Actual: {})", self.min_value, val)
                            };
                            value_message = Some(msg.clone());
                            let line = offset_to_line(m.start());
                            markers.push(ValidationMarker {
                                severity: "Warning".to_string(),
                                message: msg,
                                line,
                            });
                        }
                    }
                }
            }
        }

        // 2. Conflict item check using Aho-Corasick (O(N) search)
        if let Some(ac) = &self.conflict_items_ac {
            for mat in ac.find_iter(text) {
                let pattern_idx = mat.pattern().as_usize();
                if pattern_idx < self.conflict_item_reasons.len() {
                    let reason = &self.conflict_item_reasons[pattern_idx];
                    let msg = format!("Conflict detected: {}", reason);
                    let line = offset_to_line(mat.start());
                    markers.push(ValidationMarker {
                        severity: "Warning".to_string(),
                        message: msg,
                        line,
                    });
                }
            }
        }

        // 3. Keyword Rule Check (Dynamic)
        for rule in &self.keyword_rules {
            if !rule.trigger_words.is_empty() && rule.trigger_words.iter().all(|word| text.contains(word.as_str())) {
                has_keyword_conflict = true;
                let msg = rule.warning_message.clone();
                keyword_message = Some(msg.clone());
                // Find the line of the first trigger word.
                let line = rule.trigger_words.first()
                    .and_then(|w| text.find(w.as_str()))
                    .map(|o| offset_to_line(o))
                    .unwrap_or(1);
                markers.push(ValidationMarker {
                    severity: "Error".to_string(),
                    message: msg,
                    line,
                });
            }
        }

        let is_valid = markers.iter().all(|m| m.severity != "Error");

        ValidationResult {
            is_valid,
            extracted_value,
            has_keyword_conflict,
            keyword_message,
            value_message,
            markers,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::Profile;

    #[test]
    fn test_validation_duration_and_conflict() {
        let json_data = r#"{
            "assessment_rules": {
                "document_template": {
                    "numerical_requirements": {
                        "min_value": 50,
                        "value_regex_pattern": "Budget: ([0-9]+)K",
                        "value_warning_template": "Budget too low. Required: {min}K, Actual: {actual}K.",
                        "conflict_items": [
                            {
                                "item_code": "VEN-001",
                                "name": "Banned Vendor",
                                "reason": "This vendor is blocked by compliance."
                            }
                        ]
                    }
                },
                "keyword_rules": [
                    {
                        "trigger_words": ["Confidential", "Public Leak"],
                        "warning_message": "Confidentiality breach detected."
                    }
                ]
            }
        }"#;

        let profile = Profile::load_from_json_str(json_data).unwrap();
        let validator = Validator::new(&profile);

        // Test numerical warning
        let res1 = validator.validate("The proposed Budget: 40K is ready.");
        assert_eq!(res1.extracted_value, Some(40));
        assert!(res1.value_message.as_deref().unwrap().contains("Required: 50K"));

        // Test conflict item warning
        let res2 = validator.validate("We will use Banned Vendor for this.");
        assert!(!res2.markers.is_empty());
        assert!(res2.markers[0].message.contains("Banned Vendor"));

        // Test keyword warning
        let res3 = validator.validate("This is Confidential and causes a Public Leak.");
        assert!(!res3.is_valid);
        assert!(res3.has_keyword_conflict);
        assert!(res3.keyword_message.as_deref().unwrap().contains("breach"));
    }
}
