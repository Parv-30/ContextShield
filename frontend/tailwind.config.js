/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        primary: "#1E40AF",
        "on-primary": "#FFFFFF",
        secondary: "#3B82F6",
        accent: "#D97706",
        "on-accent": "#000000",
        background: "#F8FAFC",
        foreground: "#1E3A8A",
        card: "#FFFFFF",
        muted: "#E9EEF6",
        "muted-foreground": "#475569",
        border: "#DBEAFE",
        destructive: "#DC2626",
        "on-destructive": "#FFFFFF",
        ring: "#1E40AF",
      },
      fontFamily: {
        sans: ["Fira Sans", "system-ui", "sans-serif"],
        mono: ["Fira Code", "monospace"],
      },
      boxShadow: {
        sm: "0 1px 2px rgba(0,0,0,0.05)",
        md: "0 4px 6px rgba(0,0,0,0.1)",
        lg: "0 10px 15px rgba(0,0,0,0.1)",
      },
      borderRadius: {
        DEFAULT: "8px",
        lg: "12px",
      },
    },
  },
  plugins: [],
}
