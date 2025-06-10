# Using Registers
The `Register` class is an important component in the FlexRAG that integrates configuration files and loads various RAG components. The registrar can gather multiple components of the same type and generate a unified configuration structure to help you configure and use these components. This tutorial will show you how to use the registrar in FlexRAG.

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
If you wish to develop your project by modifying the FlexRAG source code, all registrars can be used as decorators to register new components. However, if you use the `run_assistant` or `run_interactive` entrypoints of FlexRAG, **only** the `ASSISTANTS` registrar can be used to register new components.
```

### Registering a New Component
To register a new component, simply decorate the component class with the corresponding register. For example, to register a new `Assistant` component, you can use the `ASSISTANTS` register as shown below:

```python
from flexrag.assistant import AssistantBase, ASSISTANTS
from flexrag.utils import configure

@configure
class MyAssistantConfig:
    # Define your assistant configuration here
    pass

@ASSISTANTS("my_assistant", config_class=MyAssistantConfig)
class MyAssistant(AssistantBase):
    # Define your assistant here
    def answer(self, question: str) -> str:
        return "MyAssistant: " + question
```

The register takes the following arguments, namely `shortnames` and `config_class`. 
- The `shortnames` argument is a list of shortnames of the component, which serve as simplified names for the component, making it easier to reference when loading. 
- The `config_class` argument is the configuration class for the component. This parameter is optional—if not provided, the component will not use any configuration.

### Generating the Configuration
After registering the component, you can generate the configuration `dataclass` for all the registered components using the `make_config` function. For example, to generate the configuration for all the registered `Assistant` components, you can use the `make_config` function as shown below:

```python
AssistantConfig = ASSISTANTS.make_config()
```

The generated `AssistantConfig` class will have the following structure:

```python
# configure is a special decorator that helps to define the configuration dataclass in FlexRAG
from flexrag.utils import configure

@configure
class AssistantConfig:
    # The shortname of the assistant
    assistant_type: str
    # The name of the configuration is the first shortname + "_config"
    my_assistant_config: MyAssistantConfig  
    modular_config: ModularAssistantConfig
    # Other registered assistant configurations
    ...  
```

```{tip}
In the FlexRAG entrypoints, many configurations are generated in this way. This allows us to flexibly modify the components and their configurations in the workflow through configuration files.
```

### Loading the Component
To load the component using the configuration, you can use the `load` function of the register. For example, to load the `MyAssistant` component using the configuration, you can use the `load` function as shown below:

```python
AssistantConfig.assistant_type = "my_assistant"
my_assistant = ASSISTANTS.load(AssistantConfig)
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
