.PHONY: test run

test:
	pytest

run:
	uvicorn app.main:app --reload
