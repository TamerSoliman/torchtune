# ==============================================================================
# ANNOTATED: Configuration Parsing - CLI to Config Bridge
# ==============================================================================
# Source: torchtune/config/_parse.py + torchtune/config/_utils.py
#
# **WHAT**: Parses YAML config files and CLI overrides into OmegaConf DictConfig
#
# **WHY**: Provides user-friendly interface for configuring recipes
#          - Users don't write Python code
#          - YAML files define base configurations
#          - CLI overrides allow quick experimentation
#          - All without modifying recipe Python code
#
# **HOW**: Two-stage process
#          1. Load YAML file into defaults
#          2. Merge CLI overrides with precedence
#
# **DESIGN PATTERN**: Decorator + Parser + Merger
#                     - @parse: Decorator wraps recipe main()
#                     - TuneRecipeArgumentParser: Custom argparse subclass
#                     - _merge_yaml_and_cli_args: Intelligent merge logic
#
# **WORKFLOW POSITION**: Step 1 in training pipeline
#   CLI Command → [Config Parsing] → DictConfig → instantiate() → Objects
#
# **KEY FILES**:
#   - torchtune/config/_parse.py: Decorator and parser
#   - torchtune/config/_utils.py: Merge logic
#   - torchtune/config/_instantiate.py: Next step (object creation)
# ==============================================================================

import argparse
import functools
from argparse import Namespace
from typing import Any, Callable
from omegaconf import DictConfig, OmegaConf


# ==============================================================================
# Type Alias for Recipe Functions
# ==============================================================================
# **WHAT**: Type hint for recipe main functions
#
# **SIGNATURE**: Takes DictConfig, returns Any (usually None, but sys.exit int)
#
# **EXAMPLE**:
#   def recipe_main(cfg: DictConfig) -> None:
#       model = instantiate(cfg.model)
#       ...
# ==============================================================================
Recipe = Callable[[DictConfig], Any]


# ==============================================================================
# TuneRecipeArgumentParser: Custom ArgumentParser
# ==============================================================================
# **WHAT**: Extends argparse.ArgumentParser with YAML config support
#
# **WHY**: Standard argparse only handles CLI arguments
#          We want: YAML defaults + CLI overrides
#
# **KEY FEATURE**: Builtin --config argument
#                  Loads YAML file as defaults before parsing CLI args
#
# **USAGE**:
#   parser = TuneRecipeArgumentParser()
#   args, unknown = parser.parse_known_args()
#   # args contains YAML defaults + CLI overrides
#   # unknown contains key=value strings for OmegaConf
# ==============================================================================
class TuneRecipeArgumentParser(argparse.ArgumentParser):
    """
    A helpful utility subclass of the ``argparse.ArgumentParser`` that
    adds a builtin argument "config". The config argument takes a file path to a YAML file
    and loads in argument defaults from said file. The YAML file must only contain
    argument names and their values and nothing more, it does not have to include all of the
    arguments. These values will be treated as defaults and can still be overridden from the
    command line. Everything else works the same as the base ArgumentParser and you should
    consult the docs for more info: https://docs.python.org/3/library/argparse.html.

    Note:
        This class uses "config" as a builtin argument so it is not available to use.
    """

    # ==========================================================================
    # Constructor: Add --config Argument
    # ==========================================================================
    # **WHAT**: Initializes parser with mandatory --config flag
    #
    # **WHY REQUIRED**: Torchtune philosophy: configs define everything
    #                   Prevents accidentally running with wrong settings
    #
    # **ALTERNATIVE DESIGN**: Could make optional with defaults
    #                         But explicit configs = reproducibility!
    # ==========================================================================
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        super().add_argument(
            "--config",
            type=str,
            help="Path/name of a yaml file with recipe args",
            required=True,  # ← Must provide config file!
        )

    # ==========================================================================
    # parse_known_args: Load YAML Defaults Before Parsing
    # ==========================================================================
    # **WHAT**: Overrides base method to inject YAML defaults
    #
    # **HOW IT WORKS**:
    #   1. Parse args first time (get --config path)
    #   2. Load YAML file from --config path
    #   3. Set YAML values as defaults (using set_defaults)
    #   4. Parse args second time (CLI overrides YAML defaults)
    #   5. Remove --config from final namespace (not needed anymore)
    #
    # **CLEVER TRICK**: Two-pass parsing!
    #   - First pass: Extract config file path
    #   - Set defaults: Inject YAML values
    #   - Second pass: CLI args override YAML defaults
    #
    # **EXAMPLE**:
    #   YAML file (config.yaml):
    #     batch_size: 4
    #     lr: 3e-4
    #
    #   CLI command:
    #     tune run --config config.yaml lr=1e-4
    #
    #   First parse:
    #     namespace.config = "config.yaml"
    #
    #   Load YAML:
    #     OmegaConf.load("config.yaml") → {batch_size: 4, lr: 3e-4}
    #
    #   Set defaults:
    #     self.set_defaults(batch_size=4, lr=3e-4)
    #
    #   Second parse (with lr=1e-4 as unknown arg):
    #     namespace.batch_size = 4 (from YAML default)
    #     namespace.lr = 3e-4 (from YAML default, overridden later by merge)
    #
    # **UNKNOWN ARGS**: key=value pairs for OmegaConf (handled by merge function)
    # ==========================================================================
    def parse_known_args(self, *args, **kwargs) -> tuple[Namespace, list[str]]:
        """This acts the same as the base parse_known_args but will first load in defaults from
        from the config yaml file if it is provided. The command line args will always take
        precident over the values in the config file. All other parsing method, such as parse_args,
        internally call this method so they will inherit this property too. For more info see
        the docs for the base method: https://docs.python.org/3/library/argparse.html#the-parse-args-method.
        """
        # ==============================================
        # Step 1: First Parse (Extract --config Path)
        # ==============================================
        namespace, unknown_args = super().parse_known_args(*args, **kwargs)

        # ==============================================
        # Step 2: Validate Unknown Args Format
        # ==============================================
        # **ALLOWED**: key=value (for OmegaConf dotlist)
        # **NOT ALLOWED**: --flag arguments (use YAML or key=value)
        #
        # **WHY**: Prevents confusion between argparse flags and OmegaConf overrides
        #          All customization through YAML + key=value, not --flags
        #
        # **EXAMPLE ERROR**:
        #   tune run --config cfg.yaml --batch-size 8  ✗ (flag not allowed)
        #   tune run --config cfg.yaml batch_size=8    ✓ (key=value works)
        # ==============================================
        unknown_flag_args = [arg for arg in unknown_args if arg.startswith("--")]
        if unknown_flag_args:
            raise ValueError(
                f"Additional flag arguments not supported: {unknown_flag_args}. "
                f"Please use --config or key=value overrides"
            )

        # ==============================================
        # Step 3: Load YAML Config
        # ==============================================
        # **OmegaConf.load()**: Loads YAML into DictConfig
        #   Supports:
        #     - Variable interpolation: ${other_key}
        #     - Type safety
        #     - Structured configs
        #
        # **ASSERTION**: "config" cannot be inside config file
        #   Why? Would create circular reference!
        # ==============================================
        config = OmegaConf.load(namespace.config)
        assert "config" not in config, "Cannot use 'config' within a config file"

        # ==============================================
        # Step 4: Set YAML Values as Defaults
        # ==============================================
        # **to_container(resolve=False)**: Convert to plain dict
        #   resolve=False: Keep ${interpolations} unresolved
        #                  Will resolve later after CLI merge
        #
        # **set_defaults()**: Argparse method
        #   Sets default values for arguments
        #   CLI args will override these defaults
        # ==============================================
        self.set_defaults(**OmegaConf.to_container(config, resolve=False))

        # ==============================================
        # Step 5: Second Parse (Apply Defaults + CLI)
        # ==============================================
        # **NOW**: namespace contains YAML defaults
        #          unknown_args still has key=value overrides
        #          These will be merged by _merge_yaml_and_cli_args()
        # ==============================================
        namespace, unknown_args = super().parse_known_args(*args, **kwargs)

        # ==============================================
        # Step 6: Remove --config from Final Namespace
        # ==============================================
        # **WHY**: --config is a bootstrap argument
        #          Recipe doesn't need to know which file was used
        #          Only the merged config values matter
        # ==============================================
        del namespace.config

        return namespace, unknown_args


# ==============================================================================
# @parse Decorator: Wrap Recipe Main Functions
# ==============================================================================
# **WHAT**: Decorator that handles config parsing for recipes
#
# **WHY**: Standardizes all recipe entry points
#          Every recipe just needs:
#            @parse
#            def recipe_main(cfg: DictConfig):
#                ...
#
# **HOW IT WORKS**:
#   1. Creates TuneRecipeArgumentParser
#   2. Parses YAML + CLI args
#   3. Merges into single DictConfig
#   4. Calls recipe_main(cfg)
#   5. Exits with recipe's return code
#
# **USAGE EXAMPLE**:
#   # recipes/lora_finetune_single_device.py
#   from torchtune.config import parse
#
#   @parse
#   def recipe_main(cfg: DictConfig) -> None:
#       model = instantiate(cfg.model)
#       ...
#
#   if __name__ == "__main__":
#       recipe_main()  # Decorator handles all parsing!
#
#   # User runs:
#   tune run lora_finetune_single_device --config llama/8B_lora.yaml lr=1e-4
#
# **DECORATOR PATTERN**: Wraps function to add behavior
#                        Original function signature unchanged
#                        Transparent to recipe code
# ==============================================================================
def parse(recipe_main: Recipe) -> Callable[..., Any]:
    """
    Decorator that handles parsing the config file and CLI overrides
    for a recipe. Use it on the recipe's main function.

    Args:
        recipe_main (Recipe): The main method that initializes
            and runs the recipe

    Examples:
        >>> @parse
        >>> def main(cfg: DictConfig):
        >>>     ...

        >>> # With the decorator, the parameters will be parsed into cfg when run as:
        >>> tune my_recipe --config config.yaml foo=bar

    Returns:
        Callable[..., Any]: the decorated main
    """

    # ==========================================================================
    # Wrapper Function: Replaces Original Function
    # ==========================================================================
    # **@functools.wraps**: Preserves original function metadata
    #                       - __name__
    #                       - __doc__
    #                       - Annotations
    #
    # **WHY**: So help text, introspection work correctly
    # ==========================================================================
    @functools.wraps(recipe_main)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # ======================================================================
        # Step 1: Create Parser
        # ======================================================================
        # **description**: Uses recipe's docstring for --help text
        # **RawDescriptionHelpFormatter**: Preserves docstring formatting
        # ======================================================================
        parser = TuneRecipeArgumentParser(
            description=recipe_main.__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        # ======================================================================
        # Step 2: Parse YAML + CLI Args
        # ======================================================================
        # **yaml_args**: Namespace with YAML defaults + known CLI args
        # **cli_args**: List of key=value strings (unknown to argparse)
        #
        # **EXAMPLE**:
        #   Command: tune run --config cfg.yaml model.lora_rank=16 lr=1e-4
        #
        #   yaml_args: Namespace(batch_size=4, lr=3e-4, ...)
        #   cli_args: ["model.lora_rank=16", "lr=1e-4"]
        # ======================================================================
        yaml_args, cli_args = parser.parse_known_args()

        # ======================================================================
        # Step 3: Merge into DictConfig
        # ======================================================================
        # **_merge_yaml_and_cli_args()**: Smart merge function
        #   - Handles nested keys (model.lora_rank)
        #   - Component override syntax (model=torchtune.models.llama3_8b)
        #   - Remove syntax (~key to remove)
        #   - OmegaConf interpolation
        #
        # **Returns**: Final DictConfig ready for instantiate()
        # ======================================================================
        conf = _merge_yaml_and_cli_args(yaml_args, cli_args)

        # ======================================================================
        # Step 4: Run Recipe
        # ======================================================================
        # **sys.exit()**: Recipes return int (0 = success, non-zero = error)
        #                 Propagates exit code to shell
        # ======================================================================
        import sys
        sys.exit(recipe_main(conf))

    return wrapper


# ==============================================================================
# _merge_yaml_and_cli_args: Intelligent Config Merger
# ==============================================================================
# **WHAT**: Merges YAML args and CLI overrides into single DictConfig
#
# **WHY**: CLI overrides need special handling:
#          - Nested keys: model.lora_rank=16
#          - Component shortcuts: model=torchtune.models.llama3_8b
#          - Remove operations: ~optimizer.weight_decay
#          - Type coercion: "None" → null
#
# **INPUTS**:
#   yaml_args: Namespace from argparse (YAML defaults)
#   cli_args: List of "key=value" strings
#
# **OUTPUT**: DictConfig with CLI overrides taking precedence
#
# **DESIGN PATTERN**: Merge with precedence rules
#                     CLI > YAML for conflicts
# ==============================================================================
def _merge_yaml_and_cli_args(yaml_args: Namespace, cli_args: list[str]) -> DictConfig:
    """
    Takes the direct output of argparse's parse_known_args which returns known
    args as a Namespace and unknown args as a dotlist (in our case, yaml args and
    cli args, respectively) and merges them into a single OmegaConf DictConfig.

    If a cli arg overrides a yaml arg with a _component_ field, the cli arg can
    be specified with the parent field directly, e.g., model=torchtune.models.lora_llama2_7b
    instead of model._component_=torchtune.models.lora_llama2_7b. Nested fields within the
    component should be specified with dot notation, e.g., model.lora_rank=16.

    Example:
        >>> config.yaml:
        >>>     a: 1
        >>>     b:
        >>>       _component_: torchtune.models.my_model
        >>>       c: 3

        >>> tune full_finetune --config config.yaml b=torchtune.models.other_model b.c=4
        >>> yaml_args, cli_args = parser.parse_known_args()
        >>> conf = _merge_yaml_and_cli_args(yaml_args, cli_args)
        >>> print(conf)
        >>> {"a": 1, "b": {"_component_": "torchtune.models.other_model", "c": 4}}

    Args:
        yaml_args (Namespace): Namespace containing args from yaml file, components
            should have _component_ fields
        cli_args (list[str]): list of key=value strings

    Returns:
        DictConfig: OmegaConf DictConfig containing merged args

    Raises:
        ValueError: If a cli override is not in the form of key=value
    """
    # ==========================================================================
    # Step 1: Convert Namespace to Dict
    # ==========================================================================
    # **vars()**: Extracts __dict__ from Namespace
    #   Namespace(a=1, b=2) → {"a": 1, "b": 2}
    # ==========================================================================
    yaml_kwargs = vars(yaml_args)

    # ==========================================================================
    # Step 2: Process CLI Overrides
    # ==========================================================================
    # **THREE TYPES OF CLI ARGS**:
    #   1. Remove syntax: ~key (removes from YAML config)
    #   2. Component shortcut: model=component.path (becomes model._component_=...)
    #   3. Regular override: key=value (or nested.key=value)
    # ==========================================================================
    cli_dotlist = []

    for arg in cli_args:
        # ======================================================================
        # Type 1: Remove Syntax (~key)
        # ======================================================================
        # **WHAT**: ~key removes key from YAML config
        #
        # **WHY**: Sometimes easier to remove than override
        #          Example: Remove weight_decay from optimizer
        #
        # **USAGE**:
        #   tune run --config cfg.yaml ~optimizer.weight_decay
        #
        # **RESTRICTION**: Cannot remove _component_ fields
        #                  (Would break instantiation)
        # ======================================================================
        if arg.startswith("~"):
            dotpath = arg[1:].split("=")[0]  # Extract key (after ~)

            if "_component_" in dotpath:
                raise ValueError(
                    f"Removing components from CLI is not supported: ~{dotpath}"
                )

            try:
                _remove_key_by_dotpath(yaml_kwargs, dotpath)
            except (KeyError, ValueError):
                raise ValueError(
                    f"Could not find key {dotpath} in yaml config to remove"
                ) from None
            continue

        # ======================================================================
        # Type 2 & 3: Regular Overrides (key=value)
        # ======================================================================
        # **PARSE**: Split on = to get key and value
        # ======================================================================
        try:
            k, v = arg.split("=", 1)  # Split only on first =
        except ValueError:
            raise ValueError(
                f"Command-line overrides must be in the form of key=value, got {arg}"
            ) from None

        # ======================================================================
        # Component Shortcut Logic
        # ======================================================================
        # **WHAT**: If overriding a component, allow shorthand
        #
        # **WITHOUT SHORTCUT** (verbose):
        #   model._component_=torchtune.models.llama3_8b
        #
        # **WITH SHORTCUT** (concise):
        #   model=torchtune.models.llama3_8b
        #
        # **HOW IT WORKS**:
        #   1. Check if key exists in YAML
        #   2. Check if that key has _component_ field
        #   3. If yes, append "._component_" to key
        #
        # **EXAMPLE**:
        #   YAML: model:
        #           _component_: torchtune.models.llama2_7b
        #           lora_rank: 8
        #
        #   CLI: model=torchtune.models.llama3_8b
        #
        #   Transformed to: model._component_=torchtune.models.llama3_8b
        #
        # **RESULT**: Changes component, preserves other fields (lora_rank)
        # ======================================================================
        if k in yaml_kwargs and _has_component(yaml_kwargs[k]):
            k += "._component_"

        # ======================================================================
        # Special Value Handling
        # ======================================================================
        # **None String → OmegaConf Null**:
        #   CLI: max_steps_per_epoch=None
        #   Without fix: "None" (string)
        #   With fix: null (proper null)
        #
        # **Leading Zeros (Checkpoint Format)**:
        #   CLI: max_filename=00005
        #   Without fix: 5 (integer, loses zeros)
        #   With fix: "00005" (string, preserves zeros)
        #
        # **OmegaConf YAML Tags**:
        #   !!null → null value
        #   !!str → force string type
        # ======================================================================
        if v == "None":
            v = "!!null"  # Convert "None" string to OmegaConf null

        # HACK: Force string for checkpoint format (preserve leading zeros)
        if "max_filename" in k:
            v = "!!str " + v

        cli_dotlist.append(f"{k}={v}")

    # ==========================================================================
    # Step 3: Create OmegaConf Configs
    # ==========================================================================
    # **from_dotlist()**: Parse list of key=value strings
    #   ["a=1", "b.c=2"] → {a: 1, b: {c: 2}}
    #
    # **create()**: Convert dict to DictConfig
    #   {a: 1, b: 2} → DictConfig({a: 1, b: 2})
    # ==========================================================================
    cli_conf = OmegaConf.from_dotlist(cli_dotlist)
    yaml_conf = OmegaConf.create(yaml_kwargs)

    # ==========================================================================
    # Step 4: Merge with CLI Precedence
    # ==========================================================================
    # **OmegaConf.merge()**: Deep merge with precedence
    #   - Later arguments override earlier
    #   - Nested dicts merged recursively
    #   - Lists replaced (not merged)
    #
    # **PRECEDENCE**: CLI > YAML
    #
    # **EXAMPLE**:
    #   YAML: {batch_size: 4, model: {lora_rank: 8}}
    #   CLI:  {batch_size: 2, model: {lora_rank: 16}}
    #   Result: {batch_size: 2, model: {lora_rank: 16}}
    # ==========================================================================
    return OmegaConf.merge(yaml_conf, cli_conf)


def _has_component(node) -> bool:
    """Check if a config node has a _component_ field."""
    return (OmegaConf.is_dict(node) or isinstance(node, dict)) and "_component_" in node


def _remove_key_by_dotpath(nested_dict: dict[str, Any], dotpath: str) -> None:
    """
    Removes a key specified by dotpath from a nested dict. Errors should handled by
    the calling function.

    Args:
        nested_dict (dict[str, Any]): dict to remove key from
        dotpath (str): dotpath of key to remove, e.g., "a.b.c"
    """
    path = dotpath.split(".")

    def delete_non_component(d: dict[str, Any], key: str) -> None:
        if _has_component(d[key]):
            raise ValueError(
                f"Removing components from CLI is not supported"
            )
        del d[key]

    def recurse_and_delete(d: dict[str, Any], path: list[str]) -> None:
        if len(path) == 1:
            delete_non_component(d, path[0])
        else:
            recurse_and_delete(d[path[0]], path[1:])
            if not d[path[0]]:  # If nested dict now empty, remove it
                delete_non_component(d, path[0])

    recurse_and_delete(nested_dict, path)


# ==============================================================================
# END-TO-END EXAMPLE WALKTHROUGH
# ==============================================================================
"""
**SCENARIO**: User fine-tunes Llama 3.1 8B with LoRA

**1. YAML Config File** (configs/llama3_1/8B_lora.yaml):
```yaml
model:
  _component_: torchtune.models.llama3_1.lora_llama3_1_8b
  lora_rank: 8
  lora_alpha: 16

optimizer:
  _component_: torch.optim.AdamW
  lr: 3e-4
  weight_decay: 0.01

batch_size: 4
epochs: 1
```

**2. CLI Command**:
```bash
tune run lora_finetune_single_device \
  --config llama3_1/8B_lora.yaml \
  model.lora_rank=16 \
  lr=1e-4 \
  ~optimizer.weight_decay
```

**3. Parsing Flow**:

Step 3a: TuneRecipeArgumentParser.parse_known_args()
  - First parse extracts: --config llama3_1/8B_lora.yaml
  - Loads YAML → DictConfig
  - Sets as defaults
  - Second parse with defaults
  - Result:
    yaml_args = Namespace(
        model={_component_: ..., lora_rank: 8, ...},
        optimizer={_component_: ..., lr: 3e-4, weight_decay: 0.01},
        batch_size=4,
        ...
    )
    cli_args = ["model.lora_rank=16", "lr=1e-4", "~optimizer.weight_decay"]

Step 3b: _merge_yaml_and_cli_args()
  - Process ~optimizer.weight_decay → removes from yaml_kwargs
  - Process model.lora_rank=16 → adds to cli_dotlist
  - Process lr=1e-4 → becomes optimizer.lr=1e-4 (if lr is under optimizer)
  - Create cli_conf from dotlist
  - Create yaml_conf from modified yaml_kwargs
  - Merge: yaml_conf + cli_conf → final config

**4. Final DictConfig**:
```python
{
    "model": {
        "_component_": "torchtune.models.llama3_1.lora_llama3_1_8b",
        "lora_rank": 16,  # ← Overridden by CLI
        "lora_alpha": 16
    },
    "optimizer": {
        "_component_": "torch.optim.AdamW",
        "lr": 1e-4  # ← Overridden by CLI
        # weight_decay removed by ~
    },
    "batch_size": 4,
    "epochs": 1
}
```

**5. Next Step**: Pass to instantiate()
```python
@parse
def recipe_main(cfg: DictConfig):
    # cfg is the final merged config above
    model = instantiate(cfg.model)
    # Creates lora_llama3_1_8b with lora_rank=16
    optimizer = instantiate(cfg.optimizer, model.parameters())
    # Creates AdamW with lr=1e-4, no weight_decay
    ...
```

**COMPONENT SHORTCUT EXAMPLE**:

YAML:
```yaml
model:
  _component_: torchtune.models.llama2_7b
  lora_rank: 8
```

CLI:
```bash
tune run --config cfg.yaml model=torchtune.models.llama3_8b
```

Processing:
1. k="model", v="torchtune.models.llama3_8b"
2. Check: yaml_kwargs["model"] has _component_? Yes
3. Transform: k = "model._component_"
4. Dotlist: ["model._component_=torchtune.models.llama3_8b"]
5. Merge result:
   ```
   model:
     _component_: torchtune.models.llama3_8b  # ← Changed
     lora_rank: 8  # ← Preserved!
   ```

**KEY INSIGHT**: Component shortcut changes the component
              but PRESERVES other nested fields!

**BENEFITS OF THIS DESIGN**:

✅ **Reproducibility**: YAML files capture full configuration
✅ **Experimentation**: CLI overrides for quick iteration
✅ **Simplicity**: Users never write Python code
✅ **Type Safety**: OmegaConf validates types
✅ **Flexibility**: Deep nested overrides (a.b.c.d=value)
✅ **Discoverability**: tune run recipe --config cfg.yaml --help
✅ **Composition**: Mix and match components easily

**INTEGRATION WITH OTHER COMPONENTS**:

┌─────────────────────────────────────────────────────────┐
│  User CLI Command                                       │
│  tune run recipe --config file.yaml key=value          │
└─────────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────┐
│  @parse Decorator (THIS FILE)                           │
│  - TuneRecipeArgumentParser: Parse YAML + CLI           │
│  - _merge_yaml_and_cli_args: Merge into DictConfig      │
└─────────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────┐
│  Recipe Main Function                                   │
│  def recipe_main(cfg: DictConfig):                      │
│      model = instantiate(cfg.model)  ← Next component!  │
└─────────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────┐
│  instantiate() (03_annotated_config_instantiation.py)   │
│  - Recursively creates objects from DictConfig          │
│  - Calls _component_ paths with parameters              │
└─────────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────┐
│  Model Builders (02_annotated_model_builders.py)        │
│  - lora_llama3_1_8b() called                            │
│  - Returns LoRA-wrapped model                           │
└─────────────────────────────────────────────────────────┘

**ALTERNATIVE DESIGNS CONSIDERED**:

❌ Pure Python Configs (like mmdetection):
   - Pros: Full Python expressiveness
   - Cons: Less reproducible, harder to version control

❌ JSON Configs:
   - Pros: Simple, widely supported
   - Cons: No comments, no interpolation, verbose

❌ Command-Line Only (no configs):
   - Pros: Simple for small experiments
   - Cons: Impossible for complex recipes, not reproducible

✅ YAML + OmegaConf + CLI Overrides (torchtune choice):
   - Pros: Best of all worlds!
   - Human-readable YAML
   - Powerful interpolation
   - CLI overrides for iteration
   - Type safety
   - Reproducible

**SUMMARY**:
Configuration parsing is the ENTRY POINT to torchtune's modular system.
It bridges user-friendly YAML/CLI interface with powerful Python objects.

Key innovations:
- Two-pass argparse (YAML defaults + CLI overrides)
- Component shortcut syntax (model=component.path)
- Remove syntax (~key)
- Deep nested overrides (a.b.c.d=value)
- Seamless integration with instantiate()

This is the FOUNDATION that makes torchtune's config-driven architecture
accessible and user-friendly!
"""
