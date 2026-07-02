import sitemap from '@astrojs/sitemap'
import { defineConfig } from 'astro/config'

// https://astro.build/config
export default defineConfig({
  // Deployed to this repo's GitHub Pages. Confirm the host once Pages is enabled:
  //   public github.com org  -> https://ucl.github.io
  //   UCL enterprise pages   -> https://github-pages.ucl.ac.uk
  // `base` is the repo name either way (used for all asset/link paths).
  site: 'https://ucl.github.io',
  base: '/BSP-isobenefit-qgis-plugin',
  trailingSlash: 'always',
  prefetch: true,
  integrations: [sitemap()],
})
