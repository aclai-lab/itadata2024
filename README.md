# Code example usage for "Symbolic Audio Classification via Modal Decision Tree Learning", ITADATA2024.

This repo contains two Jupyter Notebooks showcasing the proposed workflow for obtaining symbolic rules for symbolic audio classification.
The two notebooks are
- [ItaData2024_p_code.ipynb](ItaData2024_p_code.ipynb) for extracting propositional audio classification rules, via the [DecisionTree.jl](https://github.com/JuliaAI/DecisionTree.jl) package.
- [ItaData2024_m_code.ipynb](ItaData2024_m_code.ipynb) for extracting modal audio classification rules, via the [ModalDecisionTrees.jl](https://github.com/aclai-lab/ModalDecisionTrees.jl) package.
 
The code for extracting the rules from the learned trees relies [Sole.jl](https://github.com/aclai-lab/Sole.jl), an open-source framework for *symbolic machine learning*, originally designed for machine learning based on modal logics.

The package is mantained by the [ACLAI Lab](https://aclai.unife.it/en/) @ University of Ferrara.

## More on Sole
- [Sole.jl](https://github.com/aclai-lab/Sole.jl)
- [SoleLogics.jl](https://github.com/aclai-lab/SoleLogics.jl)
- [SoleData.jl](https://github.com/aclai-lab/SoleData.jl)
- [SoleFeatures.jl](https://github.com/aclai-lab/SoleFeatures.jl) 
- [SoleModels.jl](https://github.com/aclai-lab/SoleModels.jl)
- [SolePostHoc.jl](https://github.com/aclai-lab/SolePostHoc.jl)
