"""Shell completion for `sparkctl` — `eval "$(sparkctl completion zsh)"`.

No argcomplete dependency (the CLI is stdlib + PyYAML only): we emit a tiny bash/zsh function
that completes the verb list, then recipe stems for the recipe-taking verbs. The recipes dir is
baked in at generation time and globbed at TAB time, so newly-added recipes show up with no
regeneration and no per-keystroke python startup."""

# Verbs whose first positional is a recipe stem (recipes/<name>.yaml). `edit` also accepts `cluster`.
RECIPE_VERBS = ("apply", "validate", "pull", "pull-queue")

_ZSH = r'''# sparkctl zsh completion — add to ~/.zshrc:  eval "$(sparkctl completion zsh)"
# (requires `autoload -U compinit && compinit` earlier in your .zshrc)
_sparkctl() {
  local root="@ROOT@"
  local -a verbs recipes
  verbs=(@VERBS@)
  recipes=(${root}/recipes/*.yaml(N:t:r))
  if (( CURRENT == 2 )); then
    compadd -- $verbs
  else
    case ${words[2]} in
      @RECIPE_VERBS@) compadd -- $recipes ;;
      edit) compadd -- $recipes cluster ;;
    esac
  fi
}
compdef _sparkctl sparkctl
'''

_BASH = r'''# sparkctl bash completion — add to ~/.bashrc:  eval "$(sparkctl completion bash)"
_sparkctl() {
  local root="@ROOT@"
  local cur="${COMP_WORDS[COMP_CWORD]}"
  if [ "$COMP_CWORD" -eq 1 ]; then
    COMPREPLY=( $(compgen -W "@VERBS@" -- "$cur") )
    return
  fi
  local recipes
  recipes=$(cd "$root/recipes" 2>/dev/null && ls *.yaml 2>/dev/null | sed 's/\.yaml$//')
  case "${COMP_WORDS[1]}" in
    @RECIPE_VERBS@) COMPREPLY=( $(compgen -W "$recipes" -- "$cur") ) ;;
    edit) COMPREPLY=( $(compgen -W "$recipes cluster" -- "$cur") ) ;;
  esac
}
complete -F _sparkctl sparkctl
'''


def render(shell, root, verbs):
    """Return the completion script for `shell` ('bash'|'zsh') with `root` (repo path) and the
    `verbs` list baked in."""
    tmpl = _ZSH if shell == "zsh" else _BASH
    return (tmpl
            .replace("@ROOT@", str(root))
            .replace("@VERBS@", " ".join(verbs))
            .replace("@RECIPE_VERBS@", "|".join(RECIPE_VERBS)))
