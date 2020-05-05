#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `_climate_change_mitigation` package."""


import unittest
from click.testing import CliRunner

from _climate_change_mitigation import _climate_change_mitigation
from _climate_change_mitigation import cli


class Test_climate_change_mitigation(unittest.TestCase):
    """Tests for `_climate_change_mitigation` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_000_something(self):
        """Test something."""

    def test_command_line_interface(self):
        """Test the CLI."""
        runner = CliRunner()
        result = runner.invoke(cli.main)
        assert result.exit_code == 0
        assert '_climate_change_mitigation.cli.main' in result.output
        help_result = runner.invoke(cli.main, ['--help'])
        assert help_result.exit_code == 0
        assert '--help  Show this message and exit.' in help_result.output
