# RagQL

## Project Overview

RagQL is a local-first **Retrieval-Augmented Generation (RAG)** system designed for natural language Q\&A over your logs and databases. It provides a modular chat-like interface that can index and query local data sources such as log files and **SQLite `.db`** database files. With RagQL, you can ask questions about the contents of your logs or databases and get answers powered by a large language model, all while keeping your data **private** on your own machine. The system works by generating vector embeddings of your data and using those for context retrieval, then feeding relevant context to an LLM to produce answers. Its modular design makes it easy to extend (e.g. adding new data loaders or swapping components), and it supports dual LLM backends for maximum flexibility.

Importantly, RagQL supports both **local** and **remote** LLM/embedding backends. By default it favors a local setup using [Ollama](https://ollama.com/) (which runs open-source models on your machine) for embedding generation and question answering. This means you can run RagQL completely offline. Alternatively, you can integrate OpenAI's API for embeddings and/or LLM responses – useful if you prefer OpenAI's models or if you don't have a suitable local model. This dual backend support is seamless, allowing you to switch between local and cloud as needed (with a simple flag). RagQL's chat interface and CLI tools make it easy to interactively query your data or automate queries via scripts.

## Key Features

* **Local-First Operation** – Designed to run fully offline using local models via Ollama. All data stays on your machine, and embeddings can be generated with a local model, ensuring privacy (no log or DB data is sent to the cloud).
* **Dual Backend Support (Ollama & OpenAI)** – Use a local LLM or the OpenAI API as the backend provider. By default, RagQL will use an Ollama-served model if available, and you can **force** the use of OpenAI with a `--remote` flag. This gives you the choice between offline processing or OpenAI's latest models on demand.
* **Flexible Data Source Indexing** – Index various data sources, including plain text log files and **SQLite** database files (`.db`). RagQL uses modular loader components to parse content (e.g. reading database tables via pandas or splitting log files into chunks) and then builds a vector index (using FAISS) for efficient similarity search. This modular design makes it easy to add support for new file types or data sources in the future.
* **Interactive Chat Interface** – Launch an interactive REPL to have a multi-turn conversation with your data. The chat interface allows follow-up questions and retains context from previous Q\&A turns (conversation memory), enabling a more natural dialogue when exploring data. You can also enter special commands in this mode to manage configuration or data sources on the fly.
* **One-off Query Mode** – If you prefer not to use the chat interface, you can run RagQL as a one-shot CLI tool. By specifying a question along with target source files/folders in a single command, RagQL will process the query and print the answer, then exit. This is convenient for scripting or quick queries.
* **Configuration System** – Easily manage API keys and default settings through configuration files. RagQL supports a **`.env`** file for sensitive settings (like your OpenAI API key or Ollama server URL) and a **`config.json`** for persistent configuration (such as a list of data sources to index by default, or other preferences). A built-in config mode (`--configs`) allows you to add or remove indexed sources and set keys without manually editing files. These settings persist between runs, so you can "set and forget" your environment and data sources.
* **Lightweight & Extensible** – Built with Python and standard libraries/frameworks (FAISS for embeddings index, `pandas` for data handling, `argparse` for CLI, etc.), the project remains lightweight and hackable. Developers can easily extend RagQL – for example, by adding new loader modules for different file formats, or integrating alternative vector stores – thanks to its clean, modular architecture.

## Internals

```mermaid
flowchart TD
  subgraph Startup
    style Startup stroke:#a14FFF, fill:#a14FFF ,fill-opacity:0.1
    A["CLI: ragql -v --query"] --> B["Settings.load()"]
    B --> C{"config file exists?"}
    C -->|yes| D["parse JSON"]
    C -->|no| E["load env vars"]
    D --> F["initialize Settings"]
    E --> F
    A --> G["RagQL.__init__()"]
    G --> H["ensure DB schema"]
  end

  subgraph Indexing
    style Indexing stroke:#a14FFF, fill:#a14FFF ,fill-opacity:0.1
    A --> I["RagQL.build()"]
    I --> J["scan_documents()"]
    J --> K["discover docs & tables"]
    K --> L["hash & check chunks"]
    L --> M{"new chunks?"}
    M -->|no| N["skip embeddings"]
    M -->|yes| O["embed & store vectors"]
    N --> P["load FAISS index"]
    O --> P
  end

  subgraph Query
    style Query stroke:#a14FFF, fill:#a14FFF ,fill-opacity:0.1
    A --> Q["process query"]
    Q --> R["get_embeddings()"]
    R --> S["_openai_embed()"]
    S --> T["receive embedding"]
    Q --> U["faiss_search()"]
    U --> V["top_k hits"]
    V --> W["build_context()"]
    W --> X["format prompt"]
    X --> Y["call_openai_chat()"]
    Y --> Z["OpenAI API request"]
    Z --> AA["response JSON"]
    AA --> AB["print answer"]
  end

```

## Installation

**Prerequisites:** You'll need **Python 3.10+** and [Poetry](https://python-poetry.org/) (for dependency management) installed on your system (just if you want to contribute, otherwise the dependencies are only Python and the backend LLM provider of your choice). If you plan to use the local LLM mode, you should also install [Ollama](https://ollama.com/) and have it running (Ollama is available for macOS, Linux, and Windows; it provides a local API endpoint for running models). For remote mode, you'll need an OpenAI account and API key.

1. Follow these steps to install RagQL:

     **Via Poetry (recommended):**
          
     1. Clone the repo and enter it:
          ```bash
          git clone https://github.com/yourusername/ragql.git
          cd ragql
          ```
     
     2. Install dependencies:
     
          ```bash
          poetry install
          ```
     
     3. Configure your `.env` and `config.json` as described below.

     **Install via PyPI:**
     
     - RAGQL is also published on PyPI, so you can install it directly:
     
          ```bash
          pip install ragql
          ```
     
     - Or, if you're using Poetry in another project:
     
          ```bash
          poetry add ragql
          ```

2. After installing via PyPI, make sure to create a `.env` file in your working directory:

    ```bash
    # .env
    OPENAI_API_KEY=<your OpenAI key>
    OLLAMA_URL=http://localhost:11434
    ```
    
    Then you can run:
    
    ```bash
    ragql --help
    ```

3. **Configure Environment** – Create a `.env` file in the project root (or wherever you run RagQL) to store configuration like API keys. At minimum you should add:

   * `OPENAI_API_KEY=<your OpenAI key>` (if you plan to use OpenAI for embeddings or answers)
   * `OLLAMA_URL=http://localhost:11434` (or the appropriate URL if your Ollama server is running on a different port/host; default Ollama listens at 11434).

   RagQL will automatically load this `.env` file on startup. You can also configure these via environment variables directly, but using a `.env` is convenient for local development.
4. **(Optional) Configure Default Sources** – By default, RagQL will create a `rag_config.json` to persist settings. You can manually create or edit this file to specify directories or files that should be indexed on startup (and other config options). However, you can also use RagQL's interactive config commands to set this up after installation (see **Usage** below), so manual editing isn't required.
5. **Run RagQL** – You're all set! You can now run the tool via Poetry:

   ```bash
   poetry run ragql --help
   ```

   Or, if the Poetry environment is active, simply:

   ```bash
   ragql --help
   ```

   This will show the help message and verify that the installation was successful. (If RagQL was installed as a package or script, the `ragql` command should be available in your PATH.)

## Usage

RagQL can be used in multiple ways with various command-line options. Here's a comprehensive guide:

### Basic Command Structure

```bash
ragql [options] [command] [key_value]
```

### Command-Line Options

* `--help, -h` – Display help message and exit
* `--migrate` – Migrate your `config.json` to the new schema while preserving unchanged fields
* `--query QUESTION, -q QUESTION` – Run a single RAG-powered query and exit
* `--sources [SOURCES ...]` – Specify one or more folders/text files/Data.db files to index
* `--remote` – Force using OpenAI API even if OLLAMA_URL is set
* `--configs` – Enter configuration mode

### Operation Modes

1. **Interactive Chat Mode (REPL):**
   * Launch by running `ragql` with no arguments
   * Enter an interactive chat interface where you can ask questions
   * Type questions and get answers based on indexed data
   * Use special commands within chat (see Configuration Commands below)
   * Exit with `exit` or `Ctrl+C`

2. **One-off Query Mode:**
   ```bash
   ragql --query "Your question here" --sources path/to/data
   # or
   ragql -q "Your question here" --sources path/to/data
   ```
   * Ask a single question and get an answer
   * System will exit after providing the response

3. **Configuration Mode:**
   ```bash
   ragql --configs
   # or
   ragql [command] [key_value]
   ```

### Configuration Commands

The following commands can be used either in configuration mode (`--configs`) or directly as positional arguments:

* `add <path>` – Add a single file to the index configuration
* `add-folder <directory>` – Recursively add all files from a directory
* `remove <path>` – Remove a file or folder from configuration
* `list` – Display all configured source files/folders
* `set openai key <API_KEY>` – Configure your OpenAI API key
* `help` – Show available commands in config mode
* `exit` – Exit configuration mode (when in interactive mode)

### Example Commands

1. **Index and Query a Single File:**
   ```bash
   ragql --sources ~/logs/system.log -q "What errors occurred today?"
   ```

2. **Force Remote API Usage:**
   ```bash
   ragql --remote --sources data.db -q "Summarize this database"
   ```

3. **Configure Sources via Command Line:**
   ```bash
   ragql add-folder ~/project/logs
   ragql add ~/project/data/metrics.db
   ```

4. **Migrate Configuration:**
   ```bash
   ragql --migrate
   ```

5. **Interactive Chat with Multiple Sources:**
   ```bash
   ragql --sources ~/logs/ ~/databases/metrics.db
   ```

### Configuration Files

RagQL uses two main configuration files:

1. `.env` - For sensitive settings:
   ```bash
   OPENAI_API_KEY=<your OpenAI key>
   OLLAMA_URL=http://localhost:11434
   ```

2. `rag_config.json` - For persistent configuration:
   * Stores indexed sources
   * Maintains other preferences
   * Can be managed via `--configs` mode or direct commands

## Tech Stack

RagQL is built on a stack of modern tools and libraries:

* **Python 3** – The core language used for development. RagQL is written in Python, making it cross-platform and easy to extend or customize.
* **[Poetry](https://python-poetry.org/)** – Used for dependency management and packaging. This ensures a reproducible environment and easy installation of required packages.
* **[Ollama](https://ollama.com/)** – Provides the local LLM backend. Ollama is an open-source tool that allows running large language models on your own hardware via a simple API. RagQL uses Ollama to generate embeddings and responses locally when available (keeping data local).
* **OpenAI API** – The cloud alternative for embeddings and LLM responses. RagQL can call OpenAI's API (e.g. GPT-4, GPT-3.5, or Ada for embeddings) when a local model is not available or when explicitly requested. This gives access to powerful language models hosted by OpenAI.
* **FAISS** – [Facebook AI Similarity Search](https://github.com/facebookresearch/faiss) is used as the vector store for embeddings. RagQL leverages FAISS to index and search embeddings efficiently, enabling quick retrieval of relevant text chunks from your data.
* **pandas** – [pandas](https://pandas.pydata.org/) is used for data loading and manipulation, particularly for database files. For example, RagQL might use pandas to read tables from a SQLite `.db` and convert them into text or CSV for indexing.
* **python-dotenv** – [python-dotenv](https://github.com/theskumar/python-dotenv) is used to load environment variables from the `.env` file. This simplifies configuration of API keys and URLs without hardcoding them.
* **argparse** – RagQL's command-line interface is built using Python's built-in [argparse](https://docs.python.org/3/library/argparse.html) library. This powers the parsing of flags like `--sources`, `--remote`, and subcommands in config mode.

Additionally, RagQL is structured in a modular way (with separate components for CLI, configuration management, data loading, embedding generation, and storage). This design makes it easy for developers to understand and modify the codebase or integrate other tools (for example, swapping FAISS with a different vector database, or adding a new loader for a different file type).

## References

* [OpenAI API Documentation](https://platform.openai.com/docs/) – Official documentation for OpenAI's API, which RagQL can use for embeddings and chat completion.
* [Ollama Project Website](https://ollama.com/) – Information and download for Ollama, the local LLM engine used for offline mode.
* [FAISS Library (Facebook AI Research)](https://github.com/facebookresearch/faiss) – GitHub repo for FAISS, the vector similarity search library used for embedding indexing.
* [Poetry: Python Package Manager](https://python-poetry.org/) – Documentation for Poetry, used for managing RagQL's dependencies and environment.
* [Pandas Library](https://pandas.pydata.org/) – Official site for pandas, used in RagQL for data handling (especially with `.db` files).
* [python-dotenv](https://github.com/theskumar/python-dotenv) – GitHub repository for python-dotenv, which RagQL uses to manage environment variables from a file.
* [Python Argparse](https://docs.python.org/3/library/argparse.html) – Documentation for the argparse library used to build the CLI interface.

## License

RagQL version 0.3.0 and later is licensed under the
[Mozilla Public License 2.0](LICENSE). Copyright is held by Cristian Camargo
Filho personally.

The MPL requires distributed modifications to covered files to remain
available as MPL-licensed source. It permits commercial use and larger works
whose separate files use other licenses.

Releases through version 0.2.0 are not retroactively relicensed. See
[NOTICE.md](NOTICE.md) for the exact boundary.
