import sitemap from '@astrojs/sitemap'
import { defineConfig } from 'astro/config'

// https://astro.build/config
export default defineConfig({
  // Deployed to this repo's GitHub Pages. The live canonical host (confirmed by following
  // the redirect from ucl.github.io) is UCL's enterprise Pages:
  site: 'https://github-pages.ucl.ac.uk',
  base: '/BSP-isobenefit-qgis-plugin',
  trailingSlash: 'always',
  prefetch: true,
  integrations: [sitemap()],
})
