This is the documentation for the Palmer Lab ML code project. Its purpose is to provide an easy-to-follow set of guidelines for the standardisation of any code written for the project, specifically in areas where automatic formatting packages, such as Black (https://github.com/psf/black), will be unable to. The structure, and contents, of this document are derived primarily from PEP 8 (https://peps.python.org/pep-0008/), with modifications/deviations made based on lab-wide preference.

This is a living document that will update with lab and general conventions, and is open to changes and discussion from all involved parties.

## RECENT CHANGES since v0
**Current Version:** v1.0.1
Any changes since the last update will go here, for those that simply want to refresh themselves on the new stuff. As this is the first full version, everything is new though! :D

## Naming and Defining Conventions
### Names
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
print(example.addXY()) # Correct
print(example._multiplyXY()) # Wrong
```

### Functions and Methods: Annotation and General Notes
When working with methods, it is important to determine whether the method is instance-specific (that is, it will be called as a method of an _object_ of the class), class-specific (it's called by the class generically) or neither (in which case it is 'static'). Instance-specific methods should have `self` as their first argument, and `self` should then be used to refer to the object throughout:
```py
def someMethod(self, a, b):
  self._x += a
  self._y += b
```
Conversely, when working with class methods, they must first be 'decorated' by an `@classmethod` the line above the `def`, and the first argument should be `cls`. Then, as with `self` in instance-specific functions, `cls` will be used to refer to the _class_ throughout the rest of the method:
```py
@classmethod
def someClassMethod(cls):
  print(cls.object_list)
```
Class methods should be used when the function being performed is not dependent on the state of any one particular instance of the class. For example, code that allows user input to create a class instance or to manipulate a class-wide attribute (such as a counter or list of instances).

Static methods are defined as with class methods, requiring the decorator `@staticmethod` above the `def`. These methods do not take any object (either instance or class) as an argument, and so are no different to (regular) functions.
```py
@staticmethod
def someStaticMethod(string):
  print(string)
```
These can be used if a function is closely associated with class, but doesn't require object-specific attributes. It's primarily a code neatening tool, therefore, and it's inclusion is broadly up to the user (/ LAB DEBATE)

When defining functions/methods, the type of each argument should be defined with a colon and the type, such as:
```py
def someFunc(string: str, integer: int, a_list: list)
```

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
- Most importantly, ***comments should be kept up-to-date*** - any amount of commenting is useless if it contradicts the code it is supposedly explaining!
- Comments should leave a single space between the `#` and the body of the comment. Note that, for the useability of tools such as _Better Comments_, this only applies to text:
 ```py
 # Correct
 #! Also correct
 #Wrong
 #        Very wrong
 ``` 
- Comments **should be complete sentences**, and should avoid non-standard abbreviations. Extending from this, comments should be clear and easy to read - keep the tone not cringe-inducing, for example:
  ```py
  # Correct
  x, y = someFunc() # Assigns the output of some_func to x and y

  # Wrong
  x, y = someFunc() # Dude, gives the result of some_func to x and y, ya know? like, it's totes amazeballs!
  ```
  (Not going to ask for strict professionalism, because I know _I_ won't stick to that if nothing else)
- Comments **should be inline with the code (below) that they're commenting on** (at the highest level, if the code goes over multiple indents):
  ```py
  class SomeClass():

    # Here
    def someFunc():
      # Or here
      self.x = 5
    
  # But not here
      # Neither here
    def someFunc():
    # Nor even here
      self.x = 5
  ```
- Inline comments may be used, although there should be at least a single space between the end of the code and the `#`. Importantly, inline comments have a **tendency to over-explain code, or otherwise state the obvious** - this should be avoided! (At the very _least_, ensure that inline comments are constructed to act as a pseudocode for the inline comments should act in a manner similar to pseudocode):
  ```py
  x, y = someFunc() # Correct

  x, y = someFunc()# Wrong
  ```

## Docstrings
_Docstrings_ are descriptions written beneath Class and function/method definitions using `"""`, meant to describe what the code is doing, as well as its inputs and outputs. This becomes the `__doc__` attribute of a  In many IDEs (VSCode included), docstrings become dynamic elements that will be displayed when hovering over a Class or function/method.

Any public Class, function or method should have a docstring associated with it. These, ***at minimum**, must be a single line describing what it does. This line should have the `"""`s all on the same line too, and the code immediately precedes it:
```py
def add5(x):
  """A function that adds five to the input `x`, returning the result."""
  x += 5
  return x
```
However, most functions/methods, and especially classes, will require a more substantial docstring. These retain the single-line summary, but expand upon it by including a number of other sections, standardised below. It is up to the individual programmer to determine which sections they believe are important for any one docstring. The entire docstring is contained within `"""`, with the trailing `"""` on its own line at the end of the docstring:
```py
def someFunc():
  """Docstring stuff goes here
  [...]
  And still here
  """ # End of the docstring on a new line
```

All sections should start with their header on the first line, then a number of hyphens equal to the length of the header on the second, before their content starts on the third. No section is _required_, but it is advisable to at least have a **Parameters**/**Attributes** section if any arguments are taken in, and a **Returns** section if anything is returned.

The sections are as follows:
- **Description** - A longer, multi-line description of what the Class/function/method does.
- **Parameters** (for functions/methods) - A list of the arguments of a function/method, as well as their types, whether they are optional and a brief description of what they are. For optional arguments, the default should be provided in the description. For instance:
  ```py
  """
  Parameters
  ----------
  filename : str
    The filename of the .csv to be saved to.
  dataframe : DataFrame
    The dataframe to be used.
  files : list[str]
    The files to be loaded.
  verbose : bool, optional
    Whether the process should be verbose or not (default False).
  """
  ```
  Note that the type can be replaced by a set of possible values if the argument can only take very specific values:
  ```py
  """
  level_of_logging : {1, 2, 3}
    The granularity of logging to be performed, with `3` being the most granular.
  """
  ```
  And when `**kwargs` and `*args` are used, they should be listed (as `**kwargs` and `*args`) without a type. 
- **Attributes** (for a Class) - The same as **Parameters**, but for a Class definition instead.
- **Returns** - A list of datatypes, and descriptions for each, of all of the outputs of a function/method.
  ```py
  """
  Returns
  -------
  list
    A list of indices that match the search criteria.
  """
  ```
- **Raises** - A list of any errors that might arise (that aren't otherwise caught) during everyday _common_ use of the Class/function/method.
  ```py
  """
  Raises
  ------
  TypeError
    If a float is provided.
- **Notes** - Extra notes on the function, such as the theory, formulae or references.

Classes should be documented beneath the Class definition, _not_ beneath the `__init__` definition (so that the class gets assigned the `__doc__` attribute correctly).

## Bits and Bobs
- A reminder that `if x` is not the same as `if x is not None`. For instance, an empty `list`, `[]`, is not equal to `None`, for instance, but it _is_ `False`. This means that it will not trigger the first `if`, but will trigger the second.
- Never use 'bare' `except`, and minimise the amount of code in `try` clauses:
  ```py
  # Correct
  try:
    someFunc(a, b)
  except TypeError:
    pass
  
  # Wrong
  try:
    someFunc(a, b)
  except:
    pass
  ```
  Explicit catching is the preferred solution for the purpose of code readability (at a glance, one can determine the errors that are likely to arise, and so the purpose of the `try // except` is better aannotated). However, for sanity, it is acceptable to have a bare `except` _if_ the exception logged to the user:
  ```py
  except Exception as e:
    print(e)
- If a function returns anything within any of its conditions, then it should explicitly return the same number of arguments in every instance:
  ```py
  # Correct
  def someFunc(x: int):
    if x > 5:
      return x, True
    return None, False
  
  # Wrong
  def someFunc(x: int):
    if x > 5:
      return x
    return None
  
  # (Even more) Wrong
  def someFunc(x: int):
    if x > 5:
      return x
  ```
- ***Never use global variables***
