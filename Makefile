.PHONY: pretty
pretty:
	@poetry run ruff check --fix
	@poetry run ruff format