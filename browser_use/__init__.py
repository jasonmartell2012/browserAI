from browser_use.agent.service import AgentService as Agent
from browser_use.browser.service import BrowserService as Browser
from browser_use.controller.service import ControllerService as Controller
from browser_use.dom.service import DomService
from browser_use.logging_config import setup_logging

# Initialize logging when package is imported
setup_logging()

__all__ = ['Agent', 'Browser', 'Controller', 'DomService']
