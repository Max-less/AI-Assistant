import * as React from "react"
import { cn } from "@/lib/utils"

export type InputProps = React.InputHTMLAttributes<HTMLInputElement>

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, type = "text", ...props }, ref) => {
    return (
      <input
        ref={ref}
        type={type}
        className={cn(
          "w-full border-none bg-transparent font-body-primary text-body-primary text-ink placeholder:text-ink-4 focus:outline-none focus:ring-0",
          className,
        )}
        {...props}
      />
    )
  },
)
Input.displayName = "Input"

export { Input }
