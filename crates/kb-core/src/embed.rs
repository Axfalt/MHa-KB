use std::path::Path;
use std::sync::Mutex;

use anyhow::{Result, anyhow, bail};
use fastembed::{EmbeddingModel, RerankInitOptions, RerankerModel, TextEmbedding, TextInitOptions, TextRerank};

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

/// Cross-encoder rerankers: they read the question and a candidate together, so they judge
/// relevance better than vector similarity, but are too slow to run on more than a few dozen chunks
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RerankKind {
    JinaV2,
    BgeM3,
}

impl RerankKind {
    /// `None` for "none"
    pub fn parse(s: &str) -> Result<Option<Self>> {
        Ok(match s {
            "none" => None,
            "jina-v2" => Some(Self::JinaV2),
            "bge-m3" => Some(Self::BgeM3),
            other => bail!("unknown reranker '{other}' (expected none, jina-v2 or bge-m3)"),
        })
    }

    fn fastembed(self) -> RerankerModel {
        match self {
            Self::JinaV2 => RerankerModel::JINARerankerV2BaseMultilingual,
            Self::BgeM3 => RerankerModel::BGERerankerV2M3,
        }
    }
}

/// Tokens of question + chunk the reranker reads; chunks are at most ~1800 characters
const RERANK_MAX_TOKENS: usize = 512;

pub struct Embedder {
    kind: ModelKind,
    /// ONNX Runtime allows one environment per process, shared by every model
    environment: ort::environment::Environment,
    model: Mutex<TextEmbedding>,
    reranker: Option<Mutex<TextRerank>>,
}

impl Embedder {
    /// Loads the model, downloading it into `cache_dir` on first use.
    pub fn load(kind: ModelKind, cache_dir: &Path) -> Result<Self> {
        let environment = ort::init().build().map_err(|e| anyhow!("initializing ONNX Runtime: {e}"))?;
        let options = TextInitOptions::new(kind.fastembed())
            .with_environment(environment.clone())
            .with_cache_dir(cache_dir.to_path_buf())
            .with_show_download_progress(true);
        let model = TextEmbedding::try_new(options).map_err(|e| anyhow!("loading {}: {e}", kind.id()))?;
        Ok(Self { kind, environment, model: Mutex::new(model), reranker: None })
    }

    /// Adds a reranker, used by search to reorder its best candidates
    pub fn with_reranker(mut self, kind: RerankKind, cache_dir: &Path) -> Result<Self> {
        let options = RerankInitOptions::new(kind.fastembed())
            .with_environment(self.environment.clone())
            .with_cache_dir(cache_dir.to_path_buf())
            .with_max_length(RERANK_MAX_TOKENS)
            .with_show_download_progress(true);
        let model = TextRerank::try_new(options).map_err(|e| anyhow!("loading reranker {kind:?}: {e}"))?;
        self.reranker = Some(Mutex::new(model));
        Ok(self)
    }

    /// Relevance of each document to the query, in input order; `None` without a reranker
    pub fn rerank(&self, query: &str, documents: &[String]) -> Result<Option<Vec<f32>>> {
        let Some(reranker) = &self.reranker else { return Ok(None) };
        let mut model = reranker.lock().map_err(|_| anyhow!("reranker mutex poisoned"))?;
        let docs: Vec<&str> = documents.iter().map(String::as_str).collect();
        let results = model.rerank(query, docs.as_slice(), false, Some(8)).map_err(|e| anyhow!("reranking failed: {e}"))?;
        let mut scores = vec![0.0; documents.len()];
        for r in results {
            scores[r.index] = r.score;
        }
        Ok(Some(scores))
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
