/** @type {import('tailwindcss').Config} */
export default {
  content: ["./src/**/*.{html,js,ts,tsx}"],
  darkMode: 'media',
  theme: {
    extend: {
      boxShadow: {
        'inner': 'inset 0 2px 4px 0 rgb(0 0 0 / 0.05)',
      },
      fontFamily: {
        sans: ['-apple-system', 'BlinkMacSystemFont', 'Inter', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}