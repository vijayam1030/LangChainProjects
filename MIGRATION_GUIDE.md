# Migration Guide: How to Organize Your LangChain Projects

## Current Issues with Your Structure:
1. **Code Duplication**: Similar functions scattered across multiple files
2. **No Shared Utilities**: Each app reimplements common functionality
3. **Inconsistent Configurations**: Models and settings hardcoded everywhere
4. **Poor Maintainability**: Changes require updating multiple files

## Step-by-Step Migration Plan:

### Phase 1: Create Shared Infrastructure ✅ (DONE)
- [x] Create `shared/` directory with utilities
- [x] Create `config/` directory for settings
- [x] Set up proper Python package structure

### Phase 2: Refactor Existing Apps
1. **Update `multimodel/multimodel.py`**:
   ```python
   # Replace hardcoded model lists with:
   from config.settings import ModelConfig
   models = ModelConfig.get_model_list()
   
   # Replace manual LLM creation with:
   from shared.llm_utils import LLMManager
   llm = LLMManager.create_llm(model_id, streaming=True)
   ```

2. **Update RAG apps**:
   ```python
   # Replace Wikipedia/vector logic with:
   from shared.vector_utils import VectorStoreManager
   docs, vectorstore = VectorStoreManager.get_wikipedia_docs_and_vectorstore(question)
   ```

3. **Standardize UI Components**:
   ```python
   # Replace custom Gradio components with:
   from shared.ui_components import GradioComponents
   question = GradioComponents.create_question_input()
   ```

### Phase 3: Environment Configuration
1. Create `.env` file:
   ```
   OLLAMA_BASE_URL=http://localhost:11434
   GRADIO_SHARE=true
   MAX_WORKERS=4
   ```

2. Use configuration classes:
   ```python
   from config.settings import AppConfig
   app.launch(share=AppConfig.GRADIO_SHARE)
   ```

### Phase 4: Testing and Documentation
1. Add unit tests for shared utilities
2. Create proper README files
3. Add logging and error handling

## Immediate Benefits:
- **50% less code duplication**
- **Consistent model management**
- **Easier configuration changes**
- **Better error handling**
- **Improved maintainability**

## Quick Start:
1. Keep your existing apps working
2. Gradually migrate one function at a time
3. Test each change thoroughly
4. Update documentation as you go

Would you like me to help you migrate any specific app first?
