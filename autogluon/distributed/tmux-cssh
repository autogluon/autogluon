#!/bin/sh

hosts=""
ssh_options=""
tmux_session_name="cssh"

usage() {
    echo "Usage: $0 [options] host [host ...]" >&2
    echo "" >&2
    echo "Spawns multiple synchronized SSH sessions inside a tmux session." >&2
    echo "" >&2
    echo "Options:" >&2
    echo "  -h                  Show help" >&2
    echo "  -n <name>           Name of the tmux session (default: cssh)" >&2
    echo "  -o <ssh args>       Additional SSH arguments" >&2
}

while [ $# -ne 0 ]; do
    case $1 in
        -n)
            shift;
            if [ $# -eq 0 ]; then
                usage
                exit 2
            fi
            tmux_session_name="$1"; shift
            ;;
        -o)
            shift;
            if [ $# -eq 0 ]; then
                usage
                exit 2
            fi
            ssh_options="$1"; shift
            ;;
        -h)
            usage
            exit 0
            ;;
        -*)
            usage
            exit 2
            ;;
        *)
            hosts="${hosts} $1"; shift
            ;;
    esac
done

if [ -z "${hosts}" ]; then
    usage
    exit 2
fi

# Find a name for a new session
n=0; while tmux has-session -t "${tmux_session_name}-${n}" 2>/dev/null; do n=$(($n + 1)); done
tmux_session="${tmux_session_name}-${n}"

# Open a new session and split into new panes for each SSH session
for host in ${hosts}; do
    if ! tmux has-session -t "${tmux_session}" 2>/dev/null; then
        tmux new-session -s "${tmux_session}" -d "ssh ${ssh_options} ${host}"
    else
        tmux split-window -t "${tmux_session}" -d "ssh ${ssh_options} ${host}"
        # We have to reset the layout after each new pane otherwise the panes
        # quickly become too small to spawn any more
        tmux select-layout -t "${tmux_session}" tiled
    fi
done

# Synchronize panes by default
tmux set-window-option -t "${tmux_session}" synchronize-panes on

if [ -n "${TMUX}" ]; then
    # We are in a tmux, just switch to the new session
    tmux switch-client -t "${tmux_session}"
else
    # We are NOT in a tmux, attach to the new session
    tmux attach-session -t "${tmux_session}"
fi

exit 0
