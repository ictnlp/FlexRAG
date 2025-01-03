# Using Registers
FlexRAG provides a `Register` class to manage the registration of different components. The `Register` class can be instantiated to register components such as `Generator`, `Encoder`, `Retriever`, `Ranker`, `Metrics`, `Processor` and `Assistant`. This tutorial will guide you through the process of using the `Register` class to register and retrieve components.

## Using FlexRAG Registers
FlexRAG provides a set of predefined registers for different components. These registers can be used to register and retrieve components of the respective type. The following registers are available in FlexRAG:

- ASSISTANTS
- REFINERS
- CHUNKERS
- DOCUMENTPARSERS
- PROCESSORS
- METRICS
- GENERATORS
- ENCODERS
- RANKERS
- DENSE_INDEX
- RETRIEVERS
- WEB_DOWNLOADERS
- WEB_READERS

```{note}
If you are going to develop your project by modifying the FlexRAG source code or using FlexRAG as a library, all the predefined registers are available. However, if you are going to use FlexRAG's `run_assistant` or `run_interactive` entrypoints, **only** the `ASSISTANTS` register is available by default.
```

### Registering a New Component
To register a new component, simply decorate the component class with the corresponding register. For example, to register a new `Assistant` component, you can use the `ASSISTANTS` register as shown below:

```python
from dataclasses import dataclass
from flexrag.assistant import AssistantBase, ASSISTANTS

@dataclass
class MyAssistantConfig:
    # Define your assistant configuration here
    pass

@ASSISTANTS("my_assistant", config_class=MyAssistantConfig)
class MyAssistant(AssistantBase):
    # Define your assistant here
    def answer(self, question: str) -> str:
        return "MyAssistant: " + question
```

The register takes the following arguments:
*shortnames: str
    The shortnames of the component. The first shortname will be used as the default shortname.
config_class: Optional[Type]
    The configuration class for the component. If not provided, the component will not have a configuration.

### Generating the Configuration
After registering the component, you can generate the configuration `dataclass` for all the registered components using the `make_config` function. For example, to generate the configuration for all the registered `Assistant` components, you can use the `make_config` function as shown below:

```python
AssistantConfig = ASSISTANTS.make_config()
```

The generated `AssistantConfig` class will have the following structure:

```python
from dataclasses import dataclass

@dataclass
class AssistantConfig:
    # The shortname of the assistant
    assistant_type: str
    # The name of the configuration is the first shortname + "_config"
    my_assistant_config: MyAssistantConfig  
    modular_config: ModularAssistantConfig
    # Other registered assistant configurations
    ...  
```

```{note}
This step will be automatically done if you are using the `run_assistant` or `run_interactive` entrypoints.
```

### Loading the Component
To load the component using the configuration, you can use the `load` function of the register. For example, to load the `MyAssistant` component using the configuration, you can use the `load` function as shown below:

```python
AssistantConfig.assistant_type = "my_assistant"
my_assistant = ASSISTANTS.load(AssistantConfig)
```

```{note}
This step will be automatically done if you are using the `run_assistant` or `run_interactive` entrypoints.
```

## Defining a New Register
The `Register` class can be extended to define a new register for a specific component. For example, to define a new register for a `Searcher` component, you can simply create a new instance of the `Register` class as shown below:

```python
from flexrag.utils import Register

SEARCHERS = Register("searcher")
```

### Utilizing Type Hints
As the `Register` class is a generic class, you can utilize type hints to specify the type of the component that the register is managing. For example, to define a register for a `Searcher` component, you can specify the type hint as follows:

```python
from abc import ABC
from flexrag.utils import Register

class Searcher(ABC):
    pass

SEARCHERS = Register[Searcher]("searcher")
```
