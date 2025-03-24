use once_cell::sync::OnceCell;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::error::Error as StdError;

// ------------------
// Top-level JSON
// ------------------

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AiPricingJson {
    pub metered_price_id: String,
    pub providers: Vec<Provider>,
}

// ------------------
// Provider
// ------------------

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Provider {
    pub description: String,
    pub key: String,
    pub label: String,
    pub markup: Markup,
    pub models: Vec<Model>,
    pub moderation_threshold: ModerationThreshold,
    pub provider_host: String,
    pub website: String,
}

// ------------------
// Markup
// ------------------

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Markup {
    pub image_percentage: f64,
    pub text_percentage: f64,
}

// ------------------
// Moderation Threshold
// ------------------

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModerationThreshold {
    pub categories: Categories,
    pub category_score: CategoryScore,
    pub general: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Categories {
    pub hate: bool,
    #[serde(rename = "hate/threatening")]
    pub hate_threatening: bool,
    #[serde(rename = "self-harm")]
    pub self_harm: bool,
    #[serde(rename = "self-harm/instructions")]
    pub self_harm_instructions: bool,
    #[serde(rename = "self-harm/intent")]
    pub self_harm_intent: bool,
    #[serde(rename = "sexual/minors")]
    pub sexual_minors: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CategoryScore {
    #[serde(rename = "harassment/threatening")]
    pub harassment_threatening: f64,
    pub illicit: f64,
    #[serde(rename = "illicit/violent")]
    pub illicit_violent: f64,
    #[serde(rename = "violence/graphic")]
    pub violence_graphic: f64,
}

// ------------------
// Model (text/image)
// ------------------

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Model {
    pub added: String,
    pub created: String,

    #[serde(default)]
    pub features: Vec<String>,
    #[serde(default)]
    pub key: String,

    #[serde(default)]
    pub model_id: Option<String>,
    #[serde(default)]
    pub inference_profile_arn: Option<String>,
    #[serde(default)]
    pub inference_profile_id: Option<String>,

    // This can be an object (for text models) or an array (for image models).
    #[serde(default)]
    pub pricing: Option<Pricing>,

    #[serde(default)]
    pub streaming: Option<bool>,
    #[serde(default)]
    pub system_disabled: Option<bool>,

    // e.g. "text" or "image"
    #[serde(rename = "type")]
    pub model_type: String,

    #[serde(default)]
    pub deprecated: Option<bool>,
    #[serde(default)]
    pub encoder: Option<String>,

    #[serde(default)]
    pub prod_price_ids: Option<ProdPriceIds>,
}

// ------------------
// Pricing: text vs. image
// ------------------

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Pricing {
    TextPricing(TextPricing),
    ImagePricingVec(Vec<ImagePricing>),
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextPricing {
    #[serde(default)]
    pub cached_input_per1_k: Option<f64>,
    #[serde(default)]
    pub cached_input_per1_m: Option<f64>,

    pub input_per1_k: f64,
    pub input_per1_m: f64,
    pub output_per1_k: f64,
    pub output_per1_m: f64,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImagePricing {
    pub cost_per_image: f64,
    pub description: String,
    pub size: String,
}

// ------------------
// Product Price IDs
// ------------------

#[derive(Debug, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ProdPriceIds {
    #[serde(default)]
    pub cached_input: Option<String>,
    #[serde(default)]
    pub input: Option<String>,
    #[serde(default)]
    pub output: Option<String>,
}

// ------------------
// Global cache
// ------------------

static AI_PRICING: OnceCell<AiPricingJson> = OnceCell::new();

// ------------------
// Fetch function
// ------------------

/// Fetch pricing JSON from the given URL and deserialize it.
async fn fetch_pricing_json(url: &str) -> Result<AiPricingJson, Box<dyn StdError + Send + Sync>> {
    let client = Client::new();
    let resp = client.get(url).send().await?.error_for_status()?;
    let json = resp.json::<AiPricingJson>().await?;
    Ok(json)
}

/// Public function that returns the AI pricing data, with optional cache-busting.
///
/// **Important**: Because `OnceCell` is strictly synchronous, we cannot directly
/// store an `async` closure in it. Instead, we do the async work ourselves, then
/// store the result if the cell is empty.
pub async fn get_ai_pricing(
    env: &str,
    bust_cache: bool,
) -> Result<&'static AiPricingJson, Box<dyn StdError + Send + Sync>> {
    // Determine which URL to use based on environment.
    let pricing_url = if env == "prod" {
        "https://images.bookcicle.com/ai/ai-pricing.json".to_string()
    } else {
        format!("https://images.bookcicle.com/ai/ai-pricing-{}.json", env)
    };

    // If we are busting the cache, just fetch fresh data and return it
    // by leaking a Box. This won't overwrite the cell's existing value.
    if bust_cache {
        let fresh_data = fetch_pricing_json(&pricing_url).await?;
        let boxed = Box::new(fresh_data);
        let leaked_ref = Box::leak(boxed);
        return Ok(leaked_ref);
    }

    // If the cell is already set, just return a reference.
    if let Some(cached_ref) = AI_PRICING.get() {
        return Ok(cached_ref);
    }

    // Otherwise, fetch once, store in the cell, and return a reference.
    let data = fetch_pricing_json(&pricing_url).await?;
    AI_PRICING
        .set(data)
        .map_err(|_| "Cell was already initialized")?;

    // Safe to unwrap: it was just set.
    Ok(AI_PRICING.get().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::runtime::Runtime;

    #[test]
    fn test_ai_pricing_cache() {
        let rt = Runtime::new().expect("Failed to create Tokio runtime");
        rt.block_on(async {
            // Fetch from "dev" environment normally (caches result).
            let response = get_ai_pricing("dev", false)
                .await
                .expect("Failed to fetch dev environment data");
            assert!(
                !response.metered_price_id.is_empty(),
                "metered_price_id should not be empty"
            );

            let fresh = get_ai_pricing("dev", true)
                .await
                .expect("Failed to fetch dev environment data with bust_cache=true");
            assert_eq!(
                response.metered_price_id, fresh.metered_price_id,
                "Metered price IDs should match (the data is presumably the same JSON)."
            );
        });
    }
}