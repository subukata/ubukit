// Private capabilities shared by kernels and sessions; absent for the historical API.
export const SESSION_CHECKPOINT = Symbol('session checkpoint');
export const SESSION_WARM_CENTERS = Symbol('session warm centers');
export function checkpoint(options, state, origin = null) {
  options[SESSION_CHECKPOINT]?.(state, origin);
}
