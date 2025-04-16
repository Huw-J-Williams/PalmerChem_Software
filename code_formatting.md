This is the documentation for the Palmer Lab ML code project. Its purpose is to provide an easy-to-follow set of guidelines for the standardisation of any code written for the project, specifically in areas where automatic formatting packages, such as Black (https://github.com/psf/black), will be unable to. The structure, and contents, of this document are derived primarily from PEP 8 (https://peps.python.org/pep-0008/), with modifications/deviations made based on lab-wide preference. 

## Naming and Defining Conventions
### Variable Names
- **Variables** should be named in lower case, with underscores between words (known as `snake_case`).
- **Constants** (variables who's value does _not_ change throughout the running of the program) should be named in UPPER CASE, with underscores between words (as in `SCREAMING_SNAKE_CASE`).
- **Functions** should be named with `camelCase`, wherein the first letter of the first word is lower case, while the first letter of all subsequent words is capitalised, with no spaces between words. In the case of a single-word function, it will all be lower case.
- **Classes** should be named using `PascalCase`, wherein the first letter of each word is capitalised, with no paces between words.

### Single Trailing Underscores
Single leading underscores (such as in `_some_var` or `_some_func`) denote that an attribute or a method is non-public - that is, it should not be used outside of the class that owns it. Conversely, attributes and methods that lack this leading underscore are public, and so can be freely called throughout the rest of the program. As an example:
```py
class ExampleClass():

  def __init__(self, x, y):
    self._x = x
    self._y = y

    self.id = self._multiplyXY()

  def _multiplyXY(self):
    return self._x * self.y 

  def addXY(self):
    return self._x + self._y
```
In the above example, the attributes _x and _y and the method _multiplyXY() are all non-public, and so it would be **wrong** to call them directly outside of the class definition. However, the method addXY is public, and so is intended to be called elsewhere in the program:
```py
example = ExampleClass(5, 10)
print(example.addXY())
```

### Defining Functions/Methods
use self/cls
define types
avoid static methods?



## Code Layout
### Imports
- One import per line:
  ```py
  import os
  import json
  ```
  Rather than:
  ```py
  import os, json
  ```

  However, you can import multiple sub-modules from the same module:
  ```py
  from sklearn.model_selection import GridSearchCV, PredefinedSplit
  ```

### Blank Lines
- **Classes** should be surrounded by two blank lines above and below the Class block.
- **Functions** (or, within a class, Methods) should be surrounded by only a single blank line.

## Comments
- 

## Docstrings

## Bits and Bobs
- if x =/= if x is not None
- Never use bare `except`, and minimise the amount of code in `try` clauses
- Explicitly return (when something else does return). Relatedly, return the same number of objects
- ***Never use global variables***
