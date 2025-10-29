# StochasticExploratoryExperiment Documentation

This directory contains methodological documentation and design decisions for the StochasticExploratoryExperiment workflow.

## Available Documentation

### [Methodology Guide](methodology_guide.md)

Comprehensive documentation covering:

1. **Copula Methodology for Drought Return Period Analysis**
   - Why fit separate copulas for each climate scenario
   - Methodological rationale and validation
   - Common pitfalls and best practices

2. **Streamflow Scenario Comparison**
   - 3-panel figure design and interpretation
   - Design choices (linear vs log scale, fill vs line)
   - Usage and validation

3. **Centralized Styling System**
   - Color scheme rationale
   - Styling parameters and helper functions
   - Benefits and usage patterns

## Quick Links

- **For methodology questions**: See [Copula Methodology](#) section
- **For plotting questions**: See [Streamflow Comparison](#) and [Styling System](#) sections
- **For code examples**: See individual script docstrings

## Updating Documentation

This documentation should be updated when:

1. Methodological changes are made to analysis scripts
2. New visualization approaches are implemented
3. Design decisions change (e.g., color schemes, plot types)
4. New scripts are added to the workflow

## Contributing

When adding documentation:
- Use clear section headers
- Include code examples where relevant
- Explain **why** decisions were made, not just **what** was done
- Add cross-references to relevant scripts
- Update the table of contents in methodology_guide.md
