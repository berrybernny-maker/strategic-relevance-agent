from setuptools import find_packages, setup


setup(
    name="strategic-relevance-agent",
    version="0.1.0",
    description="Strategic relevance scoring agent (keyword + BM25 + optional embeddings) for market intelligence events.",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.6",
    entry_points={"console_scripts": ["sr-agent=strategic_relevance.cli:main"]},
    license="MIT",
)

