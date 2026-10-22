use std::path::Path;
use std::sync::Mutex;

use anyhow::{Result, anyhow, bail};
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};

/// Embedding models supported by the knowledge base. All are multilingual, since
/// the game text is German-keyed with EN/FR/ES translations and players ask in any of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelKind {
    E5Small,
    E5Base,
    BgeM3,
}

impl ModelKind {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "e5-small" => Self::E5Small,
            "e5-base" => Self::E5Base,
            "bge-m3" => Self::BgeM3,
            other => bail!("unknown embedding model '{other}' (expected e5-small, e5-base or bge-m3)"),
        })
    }

    pub fn id(self) -> &'static str {
        match self {
            Self::E5Small => "e5-small",
            Self::E5Base => "e5-base",
            Self::BgeM3 => "bge-m3",
        }
    }

    pub fn dims(self) -> usize {
        match self {
            Self::E5Small => 384,
            Self::E5Base => 768,
            Self::BgeM3 => 1024,
        }
    }

    fn fastembed(self) -> EmbeddingModel {
        match self {
            Self::E5Small => EmbeddingModel::MultilingualE5Small,
            Self::E5Base => EmbeddingModel::MultilingualE5Base,
            Self::BgeM3 => EmbeddingModel::BGEM3,
        }
    }

    /// (query prefix, passage prefix) expected by the model
    fn prefixes(self) -> (&'static str, &'static str) {
        match self {
            Self::E5Small | Self::E5Base => ("query: ", "passage: "),
            Self::BgeM3 => ("", ""),
        }
    }
}

pub struct Embedder {
    kind: ModelKind,
    model: Mutex<TextEmbedding>,
}

impl Embedder {
    /// Loads the model, downloading it into `cache_dir` on first use.
    pub fn load(kind: ModelKind, cache_dir: &Path) -> Result<Self> {
        let options = TextInitOptions::new(kind.fastembed())
            .with_cache_dir(cache_dir.to_path_buf())
            .with_show_download_progress(true);
        let model = TextEmbedding::try_new(options).map_err(|e| anyhow!("loading {}: {e}", kind.id()))?;
        Ok(Self { kind, model: Mutex::new(model) })
    }

    pub fn kind(&self) -> ModelKind {
        self.kind
    }

    pub fn passages(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let prefix = self.kind.prefixes().1;
        let inputs: Vec<String> = texts.iter().map(|t| format!("{prefix}{t}")).collect();
        self.embed(&inputs)
    }

    pub fn query(&self, text: &str) -> Result<Vec<f32>> {
        let prefix = self.kind.prefixes().0;
        let mut out = self.embed(&[format!("{prefix}{text}")])?;
        out.pop().ok_or_else(|| anyhow!("embedder returned no vector"))
    }

    fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut model = self.model.lock().map_err(|_| anyhow!("embedder mutex poisoned"))?;
        model.embed(inputs, Some(16)).map_err(|e| anyhow!("embedding failed: {e}"))
    }
}
