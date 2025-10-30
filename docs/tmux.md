# TMUX

## Session Management

Note that all `Ctrl-b` commands are executed within a tmux session. If you are on a cluster (HPC) and run tmux, your session remains active even if your connection to the cluster is lost or the terminal is closed. This is especially useful, for example, when you interactively allocate a GPU using `salloc --gpus=1 --partition=gpu_a100 --time=00:05:00`.

- **Create a New Session**:  
  `tmux` — Start a new session with a default name  
  `tmux new -s session_name` — Start a new session and assign it a name
  Or within an active session:  
  `:new-session -s session_name`

- **Detach from Session**:  
  `Ctrl-b d` — This detaches you from the current session, leaving it running in the background. 
  With `tmux ls`, this session will appear in the list unless it has been explicitly killed.

- **Kill a Session**:
  Simply type `exit`
  Or within an active session:
  `Ctrl-b :kill-session -t session_name`

- **Reattach to a Session**:  
  `tmux a -t session_name` — Attach to a specific session by its name.  
  You can get the session name from `tmux ls`. For example, if the output is `0: 1 windows`, the command would be `tmux a -t 0`.

  - **List Sessions**:  
  `tmux ls `  — This lists all sessions   
  `Ctrl-b s` — This lists all sessions, allowing you to switch between them. 

## Pane Management (Dividing a Window)

- **Enter copy/scroll mode**:
  `Ctrl-b [` — Then you can scroll by `↑, ↓`. To exit, press `q` or `esc`.

- **Split Window Vertically**:  
  `Ctrl-b %` — This splits the current window into two panes side by side.

- **Split Window Horizontally**:  
  `Ctrl-b "` — This splits the current window into two panes, one above the other.

- **Switch Between Panes**:  
  `Ctrl-b ←, →, ↑, ↓`  — This navigates between panes in the respective direction.
  `Ctrl-b o`  — This cycles through open panes in the current window.  
  `Ctrl-b ;`  — This toggles between the last two active panes.  

- **Resize Panes**:  
  `Ctrl-b Ctrl ←, →, ↑, ↓` — Resize pane.  
  `Ctrl-b :resize-pane -D` — Resize pane downwards.  
  `Ctrl-b :resize-pane -U` — Resize pane upwards.  
  `Ctrl-b :resize-pane -L` — Resize pane to the left.  
  `Ctrl-b :resize-pane -R` — Resize pane to the right.

- **Close a Pane**:  
  `Ctrl-b x` — This closes the current pane.

## Window (Page) Management

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

