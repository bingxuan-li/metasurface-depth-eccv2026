import type { ImgHTMLAttributes } from 'react';

type Props = ImgHTMLAttributes<HTMLImageElement> & {
  unoptimized?: boolean;
  priority?: boolean;
};

/** Static assets need no image server; retain explicit dimensions and lazy loading. */
export default function Image({ unoptimized: _unused, priority, ...props }: Props) {
  return <img {...props} fetchPriority={priority ? 'high' : undefined} />;
}
