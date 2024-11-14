import pytest

from browser_use.controller.service import Controller


def test_get_current_state():
	# Initialize controller
	controller = Controller()

	# Go to a test URL
	controller.browser.go_to_url('https://www.example.com')

	# Get current state without screenshot
	state = controller.get_state(screenshot=True)

	input('Press Enter to continue...')
