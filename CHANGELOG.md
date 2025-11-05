# [1.5.0](https://github.com/merendamattia/personal-financial-ai-agent/compare/v1.4.0...v1.5.0) (2025-11-05)


### Features

* implement caching mechanism for analyze_financial_asset ([1136a0f](https://github.com/merendamattia/personal-financial-ai-agent/commit/1136a0faceab38f1d753a858ef7b325f428ee359))

# [1.4.0](https://github.com/merendamattia/personal-financial-ai-agent/compare/v1.3.3...v1.4.0) (2025-11-05)


### Bug Fixes

* Add URL validation to prevent SSRF vulnerability ([f7e936a](https://github.com/merendamattia/personal-financial-ai-agent/commit/f7e936af5d18147e301e5c2d588feab34cf1532f))
* Address code review feedback - improve security and error handling ([66dd8ef](https://github.com/merendamattia/personal-financial-ai-agent/commit/66dd8eff9e03a4e88cbdd9130a4e8a96c298120f))


### Features

* Add API key configuration UI with settings page ([e0fa6bf](https://github.com/merendamattia/personal-financial-ai-agent/commit/e0fa6bf859f72d7dfde064a2cc80f1c5d26d7e12))
* add monthly contribution parameter to Monte Carlo simulation ([72ae5df](https://github.com/merendamattia/personal-financial-ai-agent/commit/72ae5df9d0b500b39c5e8f87a511878972fd7657))
* implement API key validation and configuration UI for providers ([4ee7a29](https://github.com/merendamattia/personal-financial-ai-agent/commit/4ee7a293f857d2b818e13d7312d2cdb133e4bdf0))

## [1.3.3](https://github.com/merendamattia/personal-financial-ai-agent/compare/v1.3.2...v1.3.3) (2025-11-04)


### Bug Fixes

* add file deletion detection and explicit analyze button ([ed1ceed](https://github.com/merendamattia/personal-financial-ai-agent/commit/ed1ceed3e8c11704a0390d391a565d4a0daa8423))

## [1.3.2](https://github.com/merendamattia/personal-financial-ai-agent/compare/v1.3.1...v1.3.2) (2025-11-03)


### Bug Fixes

* add support for SGLD asset symbol in financial analysis tool ([efd0c7e](https://github.com/merendamattia/personal-financial-ai-agent/commit/efd0c7e71515badd3e83e99122437f2fcdf0a80c))
* add unzip utility to Dockerfile for dataset extraction ([b2b5d86](https://github.com/merendamattia/personal-financial-ai-agent/commit/b2b5d86fe2844e852445ebadd4b4c07b805cf18e))
* clarify conservative allocation guidelines for investment portfolios ([7b1f4d1](https://github.com/merendamattia/personal-financial-ai-agent/commit/7b1f4d19c6f4058ecf2c8eca6f7a411b3beb0e6f))
* handle zero initial investment in wealth simulation by using a symbolic value ([b067ca9](https://github.com/merendamattia/personal-financial-ai-agent/commit/b067ca9b54d8e0cec69190a0d59b7e7ec97f44be))
* improve logging for zero initial investment in wealth simulation ([54de145](https://github.com/merendamattia/personal-financial-ai-agent/commit/54de145626937c22ee29c6952fee5304a526e764))
* update asset symbol handling in wealth simulation function ([afa2085](https://github.com/merendamattia/personal-financial-ai-agent/commit/afa2085e95bb38639a209f11e883414e26c1d865))
* update Monte Carlo simulation to use configurable initial investment value ([c38a9d3](https://github.com/merendamattia/personal-financial-ai-agent/commit/c38a9d3baeaf4681dd18856dc66563f9452a15eb))
* update portfolio justification requirements to include full asset names ([3ee53a0](https://github.com/merendamattia/personal-financial-ai-agent/commit/3ee53a0eb14bb8c7ca8b8178ebd8330e351725c1))

## [1.3.1](https://github.com/merendamattia/personal-financial-ai-agent/compare/v1.3.0...v1.3.1) (2025-10-31)


### Bug Fixes

* refine wording in financial assessment questions for clarity ([38fe997](https://github.com/merendamattia/personal-financial-ai-agent/commit/38fe997b2facea63146ab52ca3ae70868dde8fe8))
* remove assertions for occupation in financial profile tests ([b3a2180](https://github.com/merendamattia/personal-financial-ai-agent/commit/b3a2180055b4604aa97aa3c671606c7cba980a8a))
* update debug log message for financial advisor tool initialization ([3e40623](https://github.com/merendamattia/personal-financial-ai-agent/commit/3e406234166b94f007b08960050910cb65553f70))
* update financial advisor tools to reflect no available tools ([d4a975d](https://github.com/merendamattia/personal-financial-ai-agent/commit/d4a975dd7a3f6b188cf3c00e417b6befb44dbf07))
* update financial assessment questions ([ac8aa7c](https://github.com/merendamattia/personal-financial-ai-agent/commit/ac8aa7c8b10b5a8d538885fc7a983ae3643ae95f))
* update model configurations for Ollama and OpenAI ([#17](https://github.com/merendamattia/personal-financial-ai-agent/issues/17)) ([4714d66](https://github.com/merendamattia/personal-financial-ai-agent/commit/4714d664be73d1b356f2cab6bf46a821af858d4b))

# [1.3.0](https://github.com/merendamattia/personal-financial-ai-agent/compare/v1.2.0...v1.3.0) (2025-10-31)


### Bug Fixes

* correct typo ([96a1b4f](https://github.com/merendamattia/personal-financial-ai-agent/commit/96a1b4f23f2c4c25463b5b8d2a3f3f369332032d))
* correct wording in portfolio guidelines for bond allocation considerations ([a877c75](https://github.com/merendamattia/personal-financial-ai-agent/commit/a877c75159b5250b57aed75536387139ae2ed9c9))
* simplify welcome message logging by removing unnecessary state advancement ([61c002c](https://github.com/merendamattia/personal-financial-ai-agent/commit/61c002c4a0725ba8df7c52d6acd8b532d41ea367))
* update age threshold for bond allocation requirement from 40 to 45 years old ([cc479b4](https://github.com/merendamattia/personal-financial-ai-agent/commit/cc479b4ff4f96165d9f9c110e01043f10fc7a3cf))
* update age threshold for bond allocation requirement from 45/50 to 40 years old ([6bc16af](https://github.com/merendamattia/personal-financial-ai-agent/commit/6bc16afa5feda6e0a1d476ebf2a44d3b85f5b475))
* update questions for clarity and consistency in financial assessment prompts ([2d7c451](https://github.com/merendamattia/personal-financial-ai-agent/commit/2d7c451fb540f32e0fae7a880a00f4af33f8c069))


### Features

* add portfolio visualization and wealth simulation features ([#14](https://github.com/merendamattia/personal-financial-ai-agent/issues/14)) ([107508e](https://github.com/merendamattia/personal-financial-ai-agent/commit/107508e76833f4658fc41be3900baf71d4fa850a))
* enhance financial profile extraction with specific fields and defaults ([9f8fc5a](https://github.com/merendamattia/personal-financial-ai-agent/commit/9f8fc5a9e9ea4d7b53bfb7938f8254be22b8a561))
* implement PAC metrics extraction and update financial profile structure ([283bdd1](https://github.com/merendamattia/personal-financial-ai-agent/commit/283bdd1b66911677f9bba1c1811383260eb5397b))


### Performance Improvements

* optimize symbol resolution for BTC-EUR and GOLD by limiting variations ([c58bec3](https://github.com/merendamattia/personal-financial-ai-agent/commit/c58bec39c5b2cfaae71fd4d87fbabeb06b5906ff))

# [1.2.0](https://github.com/merendamattia/personal-financial-ai-agent/compare/v1.1.0...v1.2.0) (2025-10-30)


### Bug Fixes

* correct assertion syntax in TestFinancialProfileCreation ([0803e39](https://github.com/merendamattia/personal-financial-ai-agent/commit/0803e399db1dd4377d9886557555caf138f1d823))
* rename financial advisor variable for consistency in main function ([f8406d4](https://github.com/merendamattia/personal-financial-ai-agent/commit/f8406d4612eee5ea9490713eb64ab2d49df9ef7b))
* set default values for financial profile fields to improve data handling ([640ebf7](https://github.com/merendamattia/personal-financial-ai-agent/commit/640ebf7275722c51e67b9ace5453ae67f200b606))
* update default max steps for FinancialAdvisorAgent from 1 to 3 ([557b76b](https://github.com/merendamattia/personal-financial-ai-agent/commit/557b76b1cfdeee8c0362a567a3f3ffa094bdebbf))
* update default values for age_range and employment_status in FinancialProfile model ([6d01a79](https://github.com/merendamattia/personal-financial-ai-agent/commit/6d01a797c3df7b75cfc48743787276c877222b2b))
* update default years parameter in analyze_financial_asset function from 5 to 10 ([08a6121](https://github.com/merendamattia/personal-financial-ai-agent/commit/08a6121e0c426e447cc60afcf1240dc66440a0aa))


### Features

* add BaseAgent, ChatbotAgent, and FinancialAdvisorAgent modules ([2d6ff8b](https://github.com/merendamattia/personal-financial-ai-agent/commit/2d6ff8b8c907763d75ddc56a74008bd1c85e6c31))
* add financial tools for price retrieval and return calculations ([6e53f6a](https://github.com/merendamattia/personal-financial-ai-agent/commit/6e53f6a03976ec71694779f36386b6bb55ab15ba))
* add historical returns analysis for portfolio assets and enhance asset analysis tool ([eefebe9](https://github.com/merendamattia/personal-financial-ai-agent/commit/eefebe9c31ffe74ab319fde268463a870d16881c))
* enhance financial analysis tool with structured JSON response and year return models ([9dc2ed3](https://github.com/merendamattia/personal-financial-ai-agent/commit/9dc2ed3eeff3aa4a6ea4e44e705ab11de7cc23e1))
* implement financial tools with structured responses for symbol resolution, historical prices, and return calculations ([845ef28](https://github.com/merendamattia/personal-financial-ai-agent/commit/845ef2836b3be768b07e975845f6754f6f80921a))
* load financial profile from uploaded JSON file ([7bebcf4](https://github.com/merendamattia/personal-financial-ai-agent/commit/7bebcf40babacd13bd871cbcacf4baaeafd1fc5c))

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
