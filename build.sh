#!/usr/bin/env bash
# Build worksheet and slide PDFs.
#
# Usage:
#   ./build.sh              # builds the worksheets + slides listed in the DEFAULT_* arrays
#   ./build.sh source.tex   # builds just the given source (kind auto-detected)
#
# Two kinds of source, detected by content (a worksheet has a \solutions(true|false)
# toggle line; anything else is treated as a slide deck):
#   * worksheet -> <base>-sol.pdf  (solutions visible) and <base>-nosol.pdf (hidden),
#                  selected by rewriting the \solutions(true|false) line with sed.
#   * slides    -> <base>-present.pdf (\pause reveals ON) and <base>-handout.pdf
#                  (collapsed), selected by a one-line wrapper that \def-s \HANDOUT
#                  before \input-ing the deck.
# Either way, runs pdflatex -> bibtex -> pdflatex x2 whenever the source contains a
# \bibliography{...} line.
#
# Each (source, variant) pair builds concurrently, one per thread/process, in its
# own scratch directory so the parallel jobs never clobber each other's
# intermediates. Output is buffered per job and printed atomically on completion.
#
# A named single source (./build.sh foo.tex) prints any LaTeX errors inline. A
# full build (./build.sh, no args) ends with a pass/fail summary of every target;
# failing targets leave a <target>.log. Logs are overwritten each run, never
# appended — a target's .log holds only its most recent failure (and is removed
# entirely once the target builds cleanly).
#
# On success, LaTeX aux junk (.log/.aux/.bbl/.toc/.nav/... — see AUX_EXTS) is
# swept from the source dir, including foreign leftovers from manual pdflatex /
# editor runs. Set BUILD_KEEP_AUX=1 to keep them. On failure, all artifacts are
# left in place for debugging.

set -uo pipefail

DEFAULT_WORKSHEETS=(solomonoff-worksheet.tex aixi-worksheet.tex)
DEFAULT_SLIDES=(si_and_aixi_slides.tex vpg-slides.tex goalmisgen-slides.tex)

# 1 when a single source was named on the command line (errors are printed
# inline for quick debugging); 0 for a full default build (errors go to the
# per-target .log files and a pass/fail summary is printed at the end).
SINGLE=0

cd "$(dirname "$0")"

# Track buffered-output temp files so cleanup can remove them.
declare -a outfiles=()

# Nuke every temporary artifact on exit, however we exit (normal, error, or
# Ctrl-C): the per-job scratch dirs and the buffered-output temp files. The
# normal path already removes these inline; this is the backstop for interrupts.
cleanup() {
    rm -rf ./.build-* 2>/dev/null
    [ "${#outfiles[@]}" -gt 0 ] && rm -f "${outfiles[@]}" 2>/dev/null
    return 0
}
trap cleanup EXIT INT TERM

# Standard LaTeX / bibtex / beamer / latexmk intermediate extensions. All are
# regenerable and gitignored; the final .pdf and the .tex/.bib/.sty/.cls/.bst
# sources are never matched here. Builds happen in a private scratch dir, so on
# success none of these should exist in the source dir anyway — this sweep also
# clears foreign leftovers from manual pdflatex / editor runs. Skip with
# BUILD_KEEP_AUX=1 (e.g. if an editor's incremental build relies on them).
AUX_EXTS=(aux bbl blg log out toc lof lot nav snm vrb fls fdb_latexmk
          synctex.gz idx ind ilg bcf run.xml brf xdv dvi spl)
clean_aux() {
    [ -n "${BUILD_KEEP_AUX:-}" ] && return 0
    local ext
    for ext in "${AUX_EXTS[@]}"; do
        rm -f ./*."$ext" 2>/dev/null
    done
    return 0
}

# Classify a source: "worksheet" if it has a \solutions(true|false) toggle line,
# otherwise "slides" (driven by a \def\HANDOUT wrapper).
kind_of() {
    if grep -qE '^[[:space:]]*\\solutions(true|false)' "$1"; then
        echo worksheet
    else
        echo slides
    fi
}

# Build one (source, kind, variant) triple. Runs in a subshell as a background job.
# All intermediates live in a private scratch dir; only the final PDF is copied
# back to the source directory.
#   kind=worksheet -> variant in {sol,nosol}     (sed-rewrite \solutions line)
#   kind=slides    -> variant in {present,handout} (\HANDOUT wrapper around \input)
build_one() {
    local src="$1" kind="$2" variant="$3"
    local base="${src%.tex}"
    local pdf_out="${base}-${variant}.pdf"
    local log_out="${base}-${variant}.log"

    # Clear any log from a previous run up front: a log is (re)created only when
    # this build fails, so it never accumulates across runs — a stale log can't
    # outlive a now-passing target, and a failing one holds only the last errors.
    rm -f "$log_out"

    {
        echo "=========================================="
        echo "  Building $pdf_out"
        echo "=========================================="

        if [ ! -f "$src" ]; then
            echo "ERROR: $src not found" >&2
            echo "  !! build failed for $pdf_out"
            return 1
        fi

        local has_bib=0
        grep -q '^[^%]*\\bibliography{' "$src" && has_bib=1

        # Private scratch dir keeps parallel jobs from sharing .aux/.log/etc.
        local work
        work="$(mktemp -d "./.build-${base}-${variant}.XXXXXX")"
        local job="${base}-${variant}"

        # Produce the scratch source for this variant. Worksheets are rewritten
        # in place; slides get a tiny wrapper that \input-s the deck (found via
        # TEXINPUTS below) with \HANDOUT optionally defined.
        if [ "$kind" = "worksheet" ]; then
            if [ "$variant" = "sol" ]; then
                sed -E 's/^[[:space:]]*\\solutions(true|false)/\\solutionstrue/' "$src" > "$work/${job}.tex"
            else
                sed -E 's/^[[:space:]]*\\solutions(true|false)/\\solutionsfalse/' "$src" > "$work/${job}.tex"
            fi
        else
            if [ "$variant" = "present" ]; then
                printf '\\input{%s}\n'                "$base" > "$work/${job}.tex"
            else
                printf '\\def\\HANDOUT{}\\input{%s}\n' "$base" > "$work/${job}.tex"
            fi
        fi

        # Let pdflatex/bibtex find inputs (.tex deck, .bib, .sty, images) in the source dir.
        export TEXINPUTS=".:$PWD:${TEXINPUTS:-}:"
        export BIBINPUTS=".:$PWD:${BIBINPUTS:-}:"

        local ok=1
        ( cd "$work" && {
            pdflatex -interaction=nonstopmode -halt-on-error "${job}.tex" >/dev/null || exit 1
            if [ "$has_bib" = "1" ]; then
                bibtex "${job}" >/dev/null || exit 1
                pdflatex -interaction=nonstopmode -halt-on-error "${job}.tex" >/dev/null || exit 1
            fi
            pdflatex -interaction=nonstopmode -halt-on-error "${job}.tex" >/dev/null || exit 1
        } ) || ok=0

        if [ "$ok" = "1" ] && [ -f "$work/${job}.pdf" ]; then
            cp -f "$work/${job}.pdf" "$pdf_out"
            echo "  -> $pdf_out"
            rm -rf "$work"
        else
            # Preserve the log next to the source so failures stay debuggable.
            # (cp -f overwrites; combined with the rm above the log only ever
            # holds the most recent build's errors.)
            [ -f "$work/${job}.log" ] && cp -f "$work/${job}.log" "$log_out"
            if [ "$SINGLE" = "1" ]; then
                # Single named source: surface the LaTeX errors inline so the
                # user doesn't have to open the log. Show the error lines (and
                # the offending source line); fall back to the log tail.
                echo "  !! build failed for $pdf_out — LaTeX errors:" >&2
                local errs=""
                [ -f "$log_out" ] && errs="$(grep -nE '^(!|l\.[0-9]+|Runaway|Emergency stop)' "$log_out" | head -n 40)"
                if [ -n "$errs" ]; then
                    printf '%s\n' "$errs" >&2
                elif [ -f "$log_out" ]; then
                    tail -n 20 "$log_out" >&2
                fi
                echo "  (full log: $log_out)" >&2
            else
                echo "  !! build failed for $pdf_out (see $log_out)" >&2
            fi
            rm -rf "$work"
            return 1
        fi
    } 2>&1
}

# Build the job list as "src|kind|variant" entries (each yields one PDF).
declare -a jobs=()
queue() {
    local src="$1" kind
    kind="$(kind_of "$src")"
    if [ "$kind" = "worksheet" ]; then
        jobs+=("$src|worksheet|sol" "$src|worksheet|nosol")
    else
        jobs+=("$src|slides|present" "$src|slides|handout")
    fi
}

# Pick the sources: the DEFAULT_* lists, or just the one passed on the command line.
if [ $# -eq 0 ]; then
    for src in "${DEFAULT_WORKSHEETS[@]}" "${DEFAULT_SLIDES[@]}"; do queue "$src"; done
else
    SINGLE=1
    queue "$1"
fi

# Fan out: one background job per (source, variant), buffered to a temp file so the
# interleaved stdout of concurrent jobs doesn't get scrambled.
declare -a pids labels
for entry in "${jobs[@]}"; do
    IFS='|' read -r src kind variant <<< "$entry"
    out="$(mktemp)"
    build_one "$src" "$kind" "$variant" >"$out" 2>&1 &
    pids+=("$!")
    outfiles+=("$out")
    labels+=("${src%.tex}-${variant}")
done

# Reap all jobs, print their buffered output in a stable order, track failures.
rc=0
declare -a ok_list=() fail_list=()
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        ok_list+=("${labels[$i]}")
    else
        rc=1
        fail_list+=("${labels[$i]}")
    fi
    echo
    cat "${outfiles[$i]}"
    rm -f "${outfiles[$i]}"
done

# Final pass/fail summary: one line per target PDF, failures last.
echo
echo "=========================================="
echo "  Build summary (${#ok_list[@]} ok, ${#fail_list[@]} failed)"
echo "=========================================="
[ "${#ok_list[@]}"   -gt 0 ] && printf '  ok    %s\n' "${ok_list[@]}"
[ "${#fail_list[@]}" -gt 0 ] && printf '  FAIL  %s.log\n' "${fail_list[@]}"

# Sweep LaTeX aux junk from the source dir on success. On failure, leave
# everything (including any <job>.log copied out by build_one) for debugging.
if [ "$rc" = "0" ]; then
    clean_aux
fi

exit "$rc"
