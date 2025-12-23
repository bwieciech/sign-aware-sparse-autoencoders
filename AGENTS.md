# Agent Coding Guidelines
**Follow these guidelines - this is very important!**.

## Design Approach
If you are asked to implement a larger functionality, start with reasoning about the approaches that can be taken to solve the problem. If multiple approaches
can be taken, consider the pros and cons of each, and suggest which one you think is the best for the given task, along with a rough implementation plan.
Ask the user for feedback about your approach and plan before you start coding. If you are implementing a repository from scratch, confirm the language and frameworks to use.
Sketch out diagrams if you think it will be beneficial for the user to understand your reasoning. If you think you need more details from the user to ensure a reliable plan
and implementation, don't hesitate to ask them!

# General Coding Guidelines

## Structure

Follow the principles of modularity and separation of concerns. One file should typically not contain more than one class or a few closely related functions.
Each class should have a single responsibility. An exception to this is closely related dataclasses, which can be grouped together in a single file.
Create sub-packages to logically group your code as needed. For example, if you were creating a package for algebra, the structure could look something
like this, if the language of implementation is Python:
```
algebra/
    ├── __init__.py
    ├── classes/
    │   ├── __init__.py
    │   ├── matrix.py
    │   └── vector.py
    ├── operations/
    │   ├── __init__.py
    │   ├── addition.py
    │   ├── product.py
    │   ...
    ├── utils/
    │   ├── __init__.py
    │   ├── helpers.py
    │   └── validators.py
    ...
```
Note that you can use multiple levels of nesting if you think this is beneficial.

Make sure your code is modular and adheres to practices such as SOLID principles and DRY (Don't Repeat Yourself).

Get to know the existing codebase's structure before adding new modules or files. Follow the existing structure and
conventions as closely as possible.

## Testing

Test every functionality you add. After implementation, always provide a plan including:
- The list of methods/functions/classes you will be testing
- What each test will verify
Confirm the plan with the user before you start writing tests.

Use the `unittest` framework for writing tests in Python. Each test should be independent of others.

In your test, follow the `# given # when # then` convention (don't forget to
include these comments to divide your tests into the # given, # when and # then sections).
The names of your tests should describe what they are doing and what the anticipated outcome is. An example for Python is shown below:
```python
class TestArithmeticOperations(unittest.TestCase):
    # ...
    def test_add_two_numbers_returns_sum(self):
        # given
        num_a = 3
        num_b = 5

        # when
        result = add(num_a, num_b)

        # then
        self.assertEqual(result, 8)


    def test_divide_by_zero_raises_zero_division_exception(self):
        # given
        num_a = 10
        num_b = 0

        # when & then
        with self.assertRaises(ZeroDivisionError):
            _ = divide(num_a, num_b)
```
If your tests produce artifacts, make sure they are cleaned up after the test run(s).

Always run tests and make sure they pass after you have finished developing the feature you were working on.

Aim for full test coverage of your code, excluding dataclasses or scripts located under `multilayerai/scripts`. If you
are unable to achieve this, explain why to the user.

If you use an existing piece of code from the codebase, check if it is already covered by tests. If not, add extensive
tests for it.

## Comments

When adding new code, refrain from adding comments unless they **really** bring value. If code is self-explanatory,
there is no need to. Furthermore, it is not encouraged to do so! E.g. below is an example of a comment that is not needed:
```python
total_price = number_of_apples * price_per_apple  # Calculates the total price of apples
```
- Prefer explaining the "why" over the "what". If a reader can infer the behaviour from names and types, do not add an inline comment.
- Use comments to capture rationale, constraints, or non-obvious behaviour:
  - Algorithmic intent, invariants, trade-offs, or references to papers/specs.
  - Non-obvious constraints such as units, tensor shapes, tolerated ranges, or external API quirks.
  - Workarounds/hacks with links to issues/PRs and a short reason; remove them when resolved.
  - Security or performance implications that are not apparent from code alone.
- Avoid comments that restate the code, duplicate type information, or narrate each line. Remove commented-out code rather than explaining it.

Do not write docstrings. Do not start files with comments describing the file, and do not add module-level docstrings.
The class names, function names, type hints, and parameter names should tell the whole story.

Examples:
```python
# Good (cites rationale and source)
# Uses GAE to reduce variance (Schulman et al., 2016, Eq. 17).
advantages = compute_gae(...)

# Good (clarifies a non-obvious constraint)
epsilon = 1e-8  # Helps avoid log(0) when probabilities hit exact zero

# Bad (restates the code)
optimizer_step += 1  # increment counter

# Bad (narrates behaviour already obvious from names and types)
returns, advantages = compute_returns_and_advantages(...)  # Compute returns and advantages
```

### Other
Avoid long functions. If a function exceeds around 30-40 lines, consider refactoring it into smaller helper functions.

When naming variables, functions, classes, and modules, use descriptive names that convey their purpose clearly.
Avoid overly generic names like `data`, `temp`, or `var1`.

Prefer using built-in libraries and well-established third-party libraries over implementing functionality from scratch.

Prefer vectorized operations over loops for performance, especially when working with numerical data.

After finishing implementation, always review your code for potential optimizations and improvements. Remove any
unused imports, variables, or functions. Check for redundancy, unnecessary complexity, or unreachable code, and
eliminate them.

## Python

### Guidelines
Always use the virtual environment available in .venv in the project root. If no virtual environment is available, create one using
`pyenv-virtualenv`, and place it in the project's root. If you are creating a new virtual environment, list the available
Python versions and ask which one to use.

Get to know the existing codebase's coding style before adding new code. Follow the existing style and conventions as
closely as possible.

Do not use abbreviations for variable and function names. Use full words instead. For example, use `calculate_total`
instead of `calc_tot`.

**Use type hinting heavily everywhere!**

For numpy array type hints, use NDArray. For example:
```python
from numpy.typing import NDArray

def foo() -> NDArray[float]:
    return np.random.randn(3, 3, 3)
```

When rendering paths, use `os.path.join` over formatting strings:
```python
import os

path_1 = f"{root_dir}/foo/bar/baz.py"  # Not encouraged way
path_2 = os.path.join(root_dir, "foo", "bar", "baz.py")  # Encouraged way
```

If you need to install any dependencies, **you must** add them to `requirements.txt`. Refrain from installing any
dependencies in a way other than through `.venv/bin/pip install -r requirements.txt` in such a case.

Your library of choice for deep learning should be PyTorch, unless stated otherwise.
If a function or method returns multiple tensors, wrap them in a dedicated class annotated with `@tensordict.tensorclass`.
If you need to perform more granular matrix operations, use the `einops` library.

Avoid relative imports in Python. Instead, favour listing the entire package structure:
```python
from algebra.classes.matrix import Matrix  # ok
from .product import product  # not ok
```

If you ingest data that should adhere to a schema, use the `pydantic` library. This library allows defining entities,
dumping their schema, and validating JSON objects against that schema.
```python
# ...
from pydantic import BaseModel


class Gender(StrEnum):
    MALE = "MALE"
    FEMALE = "FEMALE"
    OTHER = "OTHER"


class Person(BaseModel):
    name: str
    age: int
    gender: Gender = Gender.MALE
    country: Literal["Poland", "Other"]


if __name__ == "__main__":
    person = Person(name="John Doe", age=33, country="Poland")
    print(person.model_json_schema())  # {'$defs': {'Gender': {'enum': ['MALE', 'FEMALE', 'OTHER'], 'title': 'Gender', 'type': 'string'}}, 'properties': {'name': {'title': 'Name', 'type': 'string'}, 'age': {'title': 'Age', 'type': 'integer'}, 'gender': {'$ref': '#/$defs/Gender', 'default': 'MALE'}, 'country': {'enum': ['Poland', 'Other'], 'title': 'Country', 'type': 'string'}}, 'required': ['name', 'age', 'country'], 'title': 'Person', 'type': 'object'}
    invalid_json = {"name": "Jane Doe", "age": None, "gender": "FEMALE", "country": "Germany"}
    Person.model_validate(invalid_json)
    """
    pydantic_core._pydantic_core.ValidationError: 2 validation errors for Person
    age
      Input should be a valid integer [type=int_type, input_value=None, input_type=NoneType]
    country
        Input should be 'Poland' or 'Other' [type=literal_error, input_value='Germany', input_type=str]
    """
```

Make sure your code is properly formatted. Use the `black` library if the language of implementation is Python.
To reformat a single file, run:
```bash
black <filename>
```
To reformat all files in the current directory, run:
```bash
black .
```

### Coding Style Examples
Below is an example of Python code written by the user you are helping. Do your best to write code that has a similar "look & feel" to their style.

#### Snippet #1
```python
def sample_structure(
    available_material_configurations_by_alias: Dict[str, MaterialConfiguration],
    materials_to_indices: Dict[str, int],
    chosen_materials: Collection[str],
    num_layers: int,
) -> Tuple[List[int], List[Callable[[float], complex]], List[float]]:
    output_materials = []
    output_refractive_indices = []
    output_thicknesses = []
    prev_material = None
    while len(output_materials) < num_layers:
        while (material := np.random.choice(chosen_materials)) == prev_material:
            continue
        thickness = np.random.uniform(
            available_material_configurations_by_alias[material].thickness_um_lo,
            available_material_configurations_by_alias[material].thickness_um_hi,
        )
        output_materials.append(materials_to_indices[material])
        output_refractive_indices.append(
            available_material_configurations_by_alias[
                material
            ].refractive_index_function
        )
        output_thicknesses.append(thickness)
        prev_material = material
    return output_materials, output_refractive_indices, output_thicknesses
```

#### Snippet #2
```python
class GatedSAE(SAEBase):
    def __init__(
        self,
        n_instances: int,
        d_in: int,
        d_hidden: int,
        resample_latents_threshold: int,
        device: str,
        epsilon: float = 1e-8,
    ):
        super().__init__(
            n_instances, d_in, d_hidden, resample_latents_threshold, device
        )
        self.r_magnitude = torch.nn.Parameter(
            torch.zeros(n_instances, d_hidden, device=device)
        )
        self.b_magnitude = torch.nn.Parameter(
            torch.zeros(n_instances, d_hidden, device=device)
        )
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> Dict[str, Tensor]:
        if x.ndim < 2 or x.ndim > 3:
            raise ValueError(f"Invalid dimension count of input: {x.ndim}")
        x = x.to(self.device)
        while x.ndim < 3:
            x = x.unsqueeze(0)

        W_magnitude = einops.einsum(
            self.r_magnitude.exp(),
            self.W_enc,
            "n_instances d_hidden, n_instances d_hidden d_in -> n_instances d_hidden d_in",
        )
        f_magnitude = torch.relu(
            einops.einsum(
                W_magnitude,
                x - self.b_dec.unsqueeze(1),
                "n_instances d_hidden d_in, n_instances batch d_in -> batch n_instances d_hidden",
            )
            + self.b_magnitude
        )
        # ...
        f_gate = (pi_gate > 0).float()
        latents = f_gate * f_magnitude
        W_dec_norm = self.W_dec / (self.W_dec.norm(dim=1, keepdim=True) + self.epsilon)

        # ...
        return {
            "x_hat": x_hat.transpose(0, 1),
            "pi_gate": pi_gate.transpose(0, 1),
            "x_hat_frozen": x_hat_frozen.transpose(0, 1),
            "latents": latents.transpose(0, 1),
        }
```

### Snippet #3
```python
def pad_with(
    array: NDArray[Any], left_element: Optional[Any] = None, right_element: Optional[Any] = None
) -> NDArray[Any]:
    if array.ndim == 1:
        return np.concatenate(([left_element], array, [right_element]))
    if left_element is not None:
        if isinstance(left_element, np.ndarray):
            assert left_element.ndim == 2 and left_element.shape[0] == array.shape[0]
            left_element_column = left_element
        else:
            left_element_column = np.full((array.shape[0], 1), left_element)
        array = np.hstack((left_element_column, array))
    if right_element is not None:
        if isinstance(right_element, np.ndarray):
            assert right_element.ndim == 2 and right_element.shape[0] == array.shape[0]
            right_element_column = right_element
        else:
            right_element_column = np.full((array.shape[0], 1), right_element)
        array = np.hstack((array, right_element_column))
    return array
```
