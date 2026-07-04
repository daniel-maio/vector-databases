
# --- Sentence Transformer Embedding Model --- #
EMBEDDING_MODEL="microsoft/harrier-oss-v1-0.6b"
DIM=1024

# --- Data --- #
ONBOARDING_GUIDE_PATH = "data/onboarding_guide.json"
HR_POLICIES_PATH = "data/hr_policies.json"

# --- Pinecone Index Name--- #
INDEX_NAME="human-resources-index"

## --- Namespaces --- ##
NAMESPACE_DOCUMENTS = "documents"
NAMESPACE_MEMORIES = 'conversation-history'

## --- Categories for filtering documents --- ##
FILTER_TYPE_ONBOARDING = "onboarding"
FILTER_TYPE_HR_POLICIES = "hr-policies"


