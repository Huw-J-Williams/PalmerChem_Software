This is the documentation for the Palmer Lab ML code project. Its purpose is to provide an easy-to-follow set of guidelines for the standardisation of any code written for the project, specifically in areas where automatic formatting packages, such as Black (https://github.com/psf/black), will be unable to. The structure, and contents, of this document are derived primarily from PEP 8 (), with modifications/deviations made based on lab-wide preference. 

## Naming Conventions
### Variables
- **Variables** should be named in lower case, with underscores between words (known as `snake_case`).
- **Constants** (variables who's value does _not_ change throughout the running of the program) should be named in UPPER CASE, with underscores between words (as in `SCREAMING_SNAKE_CASE`).
- **Functions** should be named with `camelCase`, wherein the first letter of the first word is lower case, while the first letter of all subsequent words is capitalised, with no spaces between words. In the case of a single-word function, it will all be lower case.
- **Classes** should be named using `PascalCase`, wherein the first letter of each word is capitalised, with no paces between words.



## Code Layout
### Blank Lines
**Classes** should be surrounded by two blank lines above and below the Class block.

**Functions** (or, within a class, Methods) should be surrounded by only a single blank line.