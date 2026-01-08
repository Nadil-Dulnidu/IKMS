/* eslint-disable @typescript-eslint/no-require-imports */
// tailwind.config.js
module.exports = {
  content: ["./src/app/**/*.{js,ts,jsx,tsx}", "./src/pages/**/*.{js,ts,jsx,tsx}", "./src/components/**/*.{js,ts,jsx,tsx}"],
  darkMode: "class",
  theme: {
    extend: {},
  },
  plugins: [require("tailwind-scrollbar")({ nocompatible: true, preferredStrategy: "pseudoelements" })],
};
