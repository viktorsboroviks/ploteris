.PHONY: \
	all \
	venv \
	format \
	format-python \
	lint \
	lint-python

all: format-python lint-python

PYTHON_VERSION := python3.12
venv: requirements.txt
	$(PYTHON_VERSION) -m venv venv
	./venv/bin/pip3 install --no-cache-dir --requirement requirements.txt

format: venv format-python

format-python: python/*.py
	. ./venv/bin/activate && \
		black $^

lint: lint-python

lint-python: python/*.py
	. ./venv/bin/activate && \
		pylint $^ ; \
		flake8 $^

clean:

distclean: clean
	rm -rfv venv
