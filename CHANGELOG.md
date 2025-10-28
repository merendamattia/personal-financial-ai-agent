# [1.1.0](https://github.com/merendamattia/personal-financial-ai-agent/compare/v1.0.0...v1.1.0) (2025-10-28)


### Bug Fixes

* improve Ollama availability check to include docker support ([3e95dae](https://github.com/merendamattia/personal-financial-ai-agent/commit/3e95dae6f13cd3ea3db3ad77272c7ac312ccb08e))
* optimize PDF chunk processing by reusing chunked text ([3bec8fc](https://github.com/merendamattia/personal-financial-ai-agent/commit/3bec8fc9677aa3bbac34cd4d8b9a3a1317260e71))
* refine logging message for localhost availability check of Ollama ([19ed449](https://github.com/merendamattia/personal-financial-ai-agent/commit/19ed449cf81cf236f1fde2f3ddd29ed443264d79))
* remove redundant numpy import in asset retriever tests ([36a59f6](https://github.com/merendamattia/personal-financial-ai-agent/commit/36a59f67f7e73a30b9227fa14763678e5e524c0e))
* remove unused tqdm import ([028ff53](https://github.com/merendamattia/personal-financial-ai-agent/commit/028ff5367fd99c3f7d3430ac82b55692805564bf))
* update .gitignore to include qdrant_storage and refine cache patterns ([d0154df](https://github.com/merendamattia/personal-financial-ai-agent/commit/d0154dfe9b3cce7af5c28bb3bf8c33ca18008933))
* update datapizza-ai-core version to 0.0.6 ([b07bbf5](https://github.com/merendamattia/personal-financial-ai-agent/commit/b07bbf5a97a13485c47c2313bd562a3b5497c7a7))
* update default number of results returned in retrieve method from 5 to 15 ([c6f52e6](https://github.com/merendamattia/personal-financial-ai-agent/commit/c6f52e6e2fdc50edeac9f6f48622130cf7c66ad6))
* update financial profile display to exclude summary notes in table ([74ae2a1](https://github.com/merendamattia/personal-financial-ai-agent/commit/74ae2a14f02ef436974ec6546154498ced1200fd))
* update global balanced allocation to include UltraShort Bond for portfolio simplification ([63a05d5](https://github.com/merendamattia/personal-financial-ai-agent/commit/63a05d5447f2c8ce9c68227a71901822ece4b440))
* update profile serialization method to model_dump for portfolio generation ([8ae9e23](https://github.com/merendamattia/personal-financial-ai-agent/commit/8ae9e234ab3bb95c2cbac4364ead89be41eecd62))
* update versions for fastembedder and qdrant ([137b2ab](https://github.com/merendamattia/personal-financial-ai-agent/commit/137b2aba663f010ed37aac282fc51c0aa78f788f))


### Features

* add Docker support with Ollama model integration ([#3](https://github.com/merendamattia/personal-financial-ai-agent/issues/3)) ([0478010](https://github.com/merendamattia/personal-financial-ai-agent/commit/0478010d9a05de808f91bcb8a248adc04475aab3))
* add geographic investment preference ([276204d](https://github.com/merendamattia/personal-financial-ai-agent/commit/276204dfaf6b5e7cfb726cacf2a533ad9a480967))
* add global balanced allocation strategy to portfolio generation prompt ([c30ffbb](https://github.com/merendamattia/personal-financial-ai-agent/commit/c30ffbbf895d6f7d45dc5be764b301183215a32a))
* add pytest configuration and initial test cases for financial profile and portfolio models ([f4b9dca](https://github.com/merendamattia/personal-financial-ai-agent/commit/f4b9dca09e10803bc7bff3321f6dcdd8046bd0b0))
* add RAG asset retriever and portfolio generation prompt for enhanced investment recommendations ([7dc2247](https://github.com/merendamattia/personal-financial-ai-agent/commit/7dc22478db265a5af37fb2bc4875734821d4751b))
* add RAG configuration for asset retriever in .env.example ([eea5a54](https://github.com/merendamattia/personal-financial-ai-agent/commit/eea5a54db7629c2b852552ec0904c3690754313d))
* enhance Dockerfile argument handling ([0d38fea](https://github.com/merendamattia/personal-financial-ai-agent/commit/0d38fea5287c34255702b42b438b85ea246f4d44))
* enhance portfolio model to support nested asset allocations and update related tests ([6d7bd9b](https://github.com/merendamattia/personal-financial-ai-agent/commit/6d7bd9b7610c5ef7fab90150d0bdebfbb9f337e8))
* implement portfolio generation and extraction features with structured response handling ([9b15ca3](https://github.com/merendamattia/personal-financial-ai-agent/commit/9b15ca322ce4dfd54fc03ab533fa7085d6876a3d))
* implement Retrieval-Augmented Generation (RAG) pipeline for document ingestion and LLM interaction ([a9f2b9c](https://github.com/merendamattia/personal-financial-ai-agent/commit/a9f2b9c4848cb73af0aa5c58a8f1df281eafed4d))
* initialize generated portfolio in session state and add portfolio generation feature ([52bc2cf](https://github.com/merendamattia/personal-financial-ai-agent/commit/52bc2cf7c684fc34c9b68e30fb1d3e867b70f240))
* streamline financial profile display with table format ([9819f29](https://github.com/merendamattia/personal-financial-ai-agent/commit/9819f29912aea464800c1179530060273f88bc91))
* update financial profile display to use DataFrame for table format ([c7b950e](https://github.com/merendamattia/personal-financial-ai-agent/commit/c7b950e625dde0a7b9a24dccc549b0a6534f3ac0))
* update investment question and summary prompt for AI-generated portfolio ([78f4b96](https://github.com/merendamattia/personal-financial-ai-agent/commit/78f4b96cfb337cfb0c9ae2c578380a6c78bd7d19))

# 1.0.0 (2025-10-25)


### Bug Fixes

* update question text to specify net monthly income for clarity ([c8bb021](https://github.com/merendamattia/personal-financial-ai-agent/commit/c8bb02139df16436d6418333a51493c7d636087a))


### Features

* add conversation completion state and update chat flow logic ([18b7327](https://github.com/merendamattia/personal-financial-ai-agent/commit/18b73279dfd75eb9c1ad674badcd6ca2cc173ad6))
* add Dockerfile and docker-compose.yml for containerized deployment ([e151c32](https://github.com/merendamattia/personal-financial-ai-agent/commit/e151c322c205acd1c0e6fde1eec6737770d7c8bb))
* add financial charts module for visualizing user financial profiles ([49ddbb6](https://github.com/merendamattia/personal-financial-ai-agent/commit/49ddbb60905c5f6b7251441bf9558499b7010044))
* add health check state management to Streamlit application ([dcb7b62](https://github.com/merendamattia/personal-financial-ai-agent/commit/dcb7b62cca9d0309494f59cfd9724ae320885422))
* add issue templates for bug reports, feature requests, and questions ([4ca3737](https://github.com/merendamattia/personal-financial-ai-agent/commit/4ca37372e90860fe9c732d50d794dfb5e841ffb0))
* add prompt for detailed financial report request ([fe388b7](https://github.com/merendamattia/personal-financial-ai-agent/commit/fe388b7c116220337d978c2766214fc6128cd6d1))
* add text streaming effect for response display in chatbot ([905ca3f](https://github.com/merendamattia/personal-financial-ai-agent/commit/905ca3fa195f27dd5fe093b4a1f308dba5bca4fa))
* enhance agent health check with task execution validation ([8bfed71](https://github.com/merendamattia/personal-financial-ai-agent/commit/8bfed7115050f14e10d1b31154b5587d8df25be0))
* enhance chatbot greeting prompt and add temperature parameter to client factories ([d67e92b](https://github.com/merendamattia/personal-financial-ai-agent/commit/d67e92bfe0815a56d3a1d68e78a1dd579d5bd3f4))
* enhance financial AI agent with multi-provider support and configuration updates ([68476aa](https://github.com/merendamattia/personal-financial-ai-agent/commit/68476aae5a9f36bb2ca1f91c1e6ae196ac54944f))
* implement financial AI agent with Streamlit interface and chatbot functionality ([8d7c61d](https://github.com/merendamattia/personal-financial-ai-agent/commit/8d7c61d2f21121f49cccec9db49256b43fa131f9))
* implement financial profile extraction and display in chatbot ([dda1925](https://github.com/merendamattia/personal-financial-ai-agent/commit/dda19256cb6d7c6c4af8ec3739f32b360d848634))
* implement question management and progress tracking in chatbot agent ([ca90d3f](https://github.com/merendamattia/personal-financial-ai-agent/commit/ca90d3fcfe9ed4773ce189e24ef009d4db4118b2))
* initialize agent state and add loading screen for provider selection ([3ed2449](https://github.com/merendamattia/personal-financial-ai-agent/commit/3ed244927b010d4186a89c4677a1143bf06ad36e))
* update financial questions for improved user engagement ([71ebb9b](https://github.com/merendamattia/personal-financial-ai-agent/commit/71ebb9b213a273cbb2547f5a8f448b41274c6142))
