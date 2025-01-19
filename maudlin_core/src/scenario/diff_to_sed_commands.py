
def diff_to_sed_commands(diff):
    sed_commands = []
    context = []
    for line in diff:
        if line.startswith('---') or line.startswith('+++') or line.startswith('@@'):
            # Reset context when encountering diff metadata
            context = []
            continue

        elif line.startswith(' '):
            # Context lines, used to determine the "range" in sed
            context.append(line[1:])

        elif line.startswith('-'):
            # Removed lines
            if context:
                # For range matching, figure out indentation from the last context line
                range_indent = " " * (len(context[-1]) - len(context[-1].lstrip()))
                # We only delete the line if it matches fully within that range
                # (Optionally, you could measure indentation from the minus line as well)
                value = line[1:].strip()
                escaped_value = (value
                                 .replace('/', '\\/')
                                 .replace('&', '\\&')
                                 .replace('[', '\\[')
                                 .replace(']', '\\]'))

                sed_commands.append(
                    f"/^{range_indent}{context[-1].strip()}/,/^{range_indent}[^ ]/"
                    f"{{/^ *{escaped_value}/d}}"
                )
            else:
                # No context? Just do a global delete
                value = line[1:].strip()
                escaped_value = (value
                                 .replace('/', '\\/')
                                 .replace('&', '\\&')
                                 .replace('[', '\\[')
                                 .replace(']', '\\]'))
                sed_commands.append(f"s/{escaped_value}/d/")

        elif line.startswith('+'):
            # Added lines
            # 1) Determine the range from the context
            if context:
                range_indent = " " * (len(context[-1]) - len(context[-1].lstrip()))
                range_pattern = context[-1].strip()

                # 2) Figure out how many spaces are in the **new line itself**
                #    so we append exactly the same indentation.
                plus_line = line[1:]  # remove the leading '+'
                new_line_indent_count = len(plus_line) - len(plus_line.lstrip())
                new_line_indent = " " * new_line_indent_count

                # 3) Escape special characters for sed
                value = plus_line.lstrip()  # actual text content, no leading spaces
                escaped_value = (value
                                 .replace('/', '\\/')
                                 .replace('&', '\\&')
                                 .replace('[', '\\[')
                                 .replace(']', '\\]'))

                # 4) Construct the sed append command
                sed_commands.append(
                    f"/^{range_indent}{range_pattern}/a\\{new_line_indent}{escaped_value}"
                )
            else:
                # No context? Just append globally (rare case)
                plus_line = line[1:]
                new_line_indent_count = len(plus_line) - len(plus_line.lstrip())
                new_line_indent = " " * new_line_indent_count

                value = plus_line.lstrip()
                escaped_value = (value
                                 .replace('/', '\\/')
                                 .replace('&', '\\&')
                                 .replace('[', '\\[')
                                 .replace(']', '\\]'))

                sed_commands.append(f"a\\{new_line_indent}{escaped_value}")

    return sed_commands
