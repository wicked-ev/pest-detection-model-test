"""
Robot state machine implementation.

Manages robot operational states and transitions:
- BOOTING: Initial startup sequence
- CHECKING_SYSTEMS: Running diagnostics
- READY: All systems OK, awaiting commands
- REMOTE_CONTROL: Responding to server commands
- AUTONOMOUS: Following predefined route
- DETECTING: Object detected, executing response
- ERROR: System failure, safe shutdown
- SHUTDOWN: Clean termination

Design patterns:
- State pattern: Each state is a distinct class
- Guard conditions: State transitions validated before occurring
- Event callbacks: State change hooks for logging/monitoring
- Safe defaults: Invalid transitions logged as warnings
"""

import logging
from enum import Enum
from typing import Optional, Callable, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from threading import Lock

logger = logging.getLogger(__name__)


class RobotState(Enum):
    """Valid robot operational states."""
    BOOTING = "BOOTING"
    WIFI_CONNECTING = "WIFI_CONNECTING"
    SERVER_CONNECTING = "SERVER_CONNECTING"
    HOTSPOT_MODE = "HOTSPOT_MODE"
    PROVISIONING = "PROVISIONING"
    CHECKING_SYSTEMS = "CHECKING_SYSTEMS"
    READY = "READY"
    REMOTE_CONTROL = "REMOTE_CONTROL"
    AUTONOMOUS = "AUTONOMOUS"
    DETECTING = "DETECTING"
    ERROR = "ERROR"
    SHUTDOWN = "SHUTDOWN"


@dataclass
class StateTransition:
    """Record of a state transition."""
    from_state: RobotState
    to_state: RobotState
    reason: str
    timestamp: datetime
    error_message: Optional[str] = None


class StateMachine:
    """
    Robot state machine with strict state transitions.
    
    Features:
    - Thread-safe state changes
    - Transition guards (prevent invalid transitions)
    - State change callbacks for event handling
    - Transition history for debugging
    - Error state with retry capability
    
    Typical flow:
        BOOTING → CHECKING_SYSTEMS → READY → (REMOTE_CONTROL | AUTONOMOUS)
    
    From any state, can enter ERROR or SHUTDOWN.
    """

    # Define valid state transitions as a graph
    VALID_TRANSITIONS: Dict[RobotState, List[RobotState]] = {
        RobotState.BOOTING: [
            RobotState.WIFI_CONNECTING,
            RobotState.HOTSPOT_MODE,
            RobotState.ERROR,
            RobotState.SHUTDOWN,
        ],
        RobotState.WIFI_CONNECTING: [
            RobotState.SERVER_CONNECTING,
            RobotState.HOTSPOT_MODE,
            RobotState.ERROR,
            RobotState.SHUTDOWN,
        ],
        RobotState.SERVER_CONNECTING: [
            RobotState.CHECKING_SYSTEMS,
            RobotState.ERROR,
            RobotState.SHUTDOWN,
        ],
        RobotState.HOTSPOT_MODE: [
            RobotState.PROVISIONING,
            RobotState.ERROR,
            RobotState.SHUTDOWN,
        ],
        RobotState.PROVISIONING: [
            RobotState.WIFI_CONNECTING,
            RobotState.ERROR,
            RobotState.SHUTDOWN,
        ],
        RobotState.CHECKING_SYSTEMS: [
            RobotState.READY,
            RobotState.ERROR,
            RobotState.SHUTDOWN,
        ],
        RobotState.READY: [
            RobotState.REMOTE_CONTROL,
            RobotState.AUTONOMOUS,
            RobotState.ERROR,
            RobotState.SHUTDOWN,
        ],
        RobotState.REMOTE_CONTROL: [
            RobotState.READY,
            RobotState.DETECTING,
            RobotState.ERROR,
            RobotState.SHUTDOWN,
        ],
        RobotState.AUTONOMOUS: [
            RobotState.READY,
            RobotState.DETECTING,
            RobotState.ERROR,
            RobotState.SHUTDOWN,
        ],
        RobotState.DETECTING: [
            RobotState.READY,
            RobotState.REMOTE_CONTROL,
            RobotState.AUTONOMOUS,
            RobotState.ERROR,
            RobotState.SHUTDOWN,
        ],
        RobotState.ERROR: [
            RobotState.WIFI_CONNECTING,
            RobotState.CHECKING_SYSTEMS,
            RobotState.SHUTDOWN,
        ],
        RobotState.SHUTDOWN: [],  # Terminal state
    }

    def __init__(self):
        """Initialize state machine in BOOTING state."""
        self._current_state = RobotState.BOOTING
        self._previous_state: Optional[RobotState] = None
        self._state_lock = Lock()
        self._error_message: Optional[str] = None
        
        # Callbacks for state changes
        self._on_state_change_callbacks: List[
            Callable[[RobotState, RobotState, str], None]
        ] = []
        
        # Transition history for debugging
        self._transition_history: List[StateTransition] = []
        self._max_history_size = 100

        logger.info(f"State machine initialized in {self._current_state.value}")

    def get_current_state(self) -> RobotState:
        """Get current robot state."""
        with self._state_lock:
            return self._current_state

    def get_previous_state(self) -> Optional[RobotState]:
        """Get previous robot state before most recent transition."""
        with self._state_lock:
            return self._previous_state

    def get_error_message(self) -> Optional[str]:
        """Get error message if in ERROR state."""
        with self._state_lock:
            return self._error_message

    def is_in_state(self, state: RobotState) -> bool:
        """Check if robot is in a specific state."""
        return self.get_current_state() == state

    def is_ready(self) -> bool:
        """Check if robot is ready for commands (READY or derived active state)."""
        current = self.get_current_state()
        return current in [
            RobotState.READY,
            RobotState.REMOTE_CONTROL,
            RobotState.AUTONOMOUS,
        ]

    def can_move(self) -> bool:
        """Check if robot can execute movement commands."""
        current = self.get_current_state()
        return current in [
            RobotState.REMOTE_CONTROL,
            RobotState.AUTONOMOUS,
            RobotState.DETECTING,
        ]

    def transition_to(
        self,
        new_state: RobotState,
        reason: str = "",
        error_message: Optional[str] = None,
    ) -> bool:
        """
        Attempt to transition to a new state.
        
        Args:
            new_state: Target state
            reason: Human-readable reason for transition
            error_message: Error details if transitioning to ERROR state
            
        Returns:
            True if transition succeeded, False if invalid
        """
        with self._state_lock:
            current = self._current_state

            # Check if transition is valid
            if new_state not in self.VALID_TRANSITIONS.get(current, []):
                logger.warning(
                    f"Invalid state transition: {current.value} → {new_state.value}"
                )
                return False

            # Perform transition
            self._previous_state = current
            self._current_state = new_state
            self._error_message = error_message

            # Record transition
            transition = StateTransition(
                from_state=current,
                to_state=new_state,
                reason=reason,
                timestamp=datetime.now(),
                error_message=error_message,
            )
            self._transition_history.append(transition)

            # Trim history if needed
            if len(self._transition_history) > self._max_history_size:
                self._transition_history.pop(0)

            logger.info(
                f"State transition: {current.value} → {new_state.value} "
                f"({reason})"
            )

        # Trigger callbacks (outside lock to avoid deadlock)
        self._trigger_callbacks(current, new_state, reason)

        return True

    def register_state_change_callback(
        self,
        callback: Callable[[RobotState, RobotState, str], None],
    ) -> None:
        """
        Register a callback to be called on state changes.
        
        Callback signature: callback(from_state, to_state, reason)
        
        Use case: Services can listen for state transitions they care about:
        - Detection service enables on AUTONOMOUS transition
        - Movement service resets on READY transition
        """
        self._on_state_change_callbacks.append(callback)
        logger.debug(f"Registered state change callback: {callback.__name__}")

    def _trigger_callbacks(
        self,
        from_state: RobotState,
        to_state: RobotState,
        reason: str,
    ) -> None:
        """Invoke all registered state change callbacks."""
        for callback in self._on_state_change_callbacks:
            try:
                callback(from_state, to_state, reason)
            except Exception as e:
                logger.error(f"State change callback failed: {e}")

    def get_transition_history(self, limit: int = 20) -> List[StateTransition]:
        """
        Get recent state transitions for debugging.
        
        Args:
            limit: Maximum number of transitions to return
            
        Returns:
            List of recent transitions, newest first
        """
        with self._state_lock:
            return self._transition_history[-limit:][::-1]

    def reset(self) -> None:
        """Reset state machine to BOOTING state."""
        with self._state_lock:
            self._current_state = RobotState.BOOTING
            self._previous_state = None
            self._error_message = None
            logger.info("State machine reset to BOOTING")

    def __str__(self) -> str:
        """Human-readable state representation."""
        current = self.get_current_state()
        if current == RobotState.ERROR:
            error_msg = self.get_error_message() or "Unknown error"
            return f"{current.value}: {error_msg}"
        return current.value


class StateTransitionError(Exception):
    """Raised when invalid state transition is attempted."""
    pass
