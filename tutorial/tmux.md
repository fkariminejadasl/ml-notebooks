# Basic Session Management

- **Create a New Session**:  
  `tmux new -s session_name`  
  Or within an active session:  
  `:new-session -s session_name`

# Window (Page) Management

- **Create a New Window (Page)**:  
  `Ctrl-b c`

- **Switch Between Windows (Pages)**:  
  `Ctrl-b n`  — This moves to the next window.  
  `Ctrl-b p`  — This moves to the previous window.  
  `Ctrl-b w`  — This lists all windows and allows you to select one to switch to.  
  `Ctrl-b <window number>`  — This directly switches to the window by its number.

- **Rename a Window**:  
  `Ctrl-b ,` — This allows you to rename the current window.

- **Close a Window (Page)**:  
  `Ctrl-b &` — This closes the current window.

# Pane Management (Dividing a Window)

- **Split Window Horizontally**:  
  `Ctrl-b %` — This splits the current window into two panes side by side.

- **Split Window Vertically**:  
  `Ctrl-b "` — This splits the current window into two panes, one above the other.

- **Switch Between Panes**:  
  `Ctrl-b o`  — This cycles through open panes in the current window.  
  `Ctrl-b ;`  — This toggles between the last two active panes.  
  `Ctrl-b ←, →, ↑, ↓`  — This navigates between panes in the respective direction.

- **Resize Panes**:  
  `Ctrl-b :resize-pane -D` — Resize pane downwards.  
  `Ctrl-b :resize-pane -U` — Resize pane upwards.  
  `Ctrl-b :resize-pane -L` — Resize pane to the left.  
  `Ctrl-b :resize-pane -R` — Resize pane to the right.

- **Close a Pane**:  
  `Ctrl-b x` — This closes the current pane.

# Session Management

- **Detach from Session**:  
  `Ctrl-b d` — This detaches you from the current session, leaving it running in the background.

- **Reattach to a Session**:  
  `tmux attach -t session_name`

- **List Sessions**:  
  `tmux ls `  — This lists all sessions   
  `Ctrl-b s` — This lists all sessions, allowing you to switch between them. 

- **Kill a Session**:  
  `Ctrl-b :kill-session -t session_name`
