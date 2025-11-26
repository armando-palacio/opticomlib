import functools
import logging
import threading


logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.basicConfig(format="%(asctime)s -- %(levelname)7s -- %(message)s", datefmt="%H:%M:%S")


class HierLogger:
    """Logger jerárquico con control de indentación automática."""

    INDENT_STR = "|   "

    def __init__(self, name: str = "hier_logger"):
        self._local = threading.local()
        self.logger = logging.getLogger(name)

    # ------------------------------------------------------------------
    # utilidades internas
    def _ensure_local_state(self):
        if not hasattr(self._local, "indent"):
            self._local.indent = 0
        if not hasattr(self._local, "suppress"):
            self._local.suppress = 0

    def indent(self):
        """Context manager que incrementa y revierte la indentación."""
        self._ensure_local_state()

        class _IndentCtx:
            def __init__(self, outer):
                self.outer = outer

            def __enter__(self):
                self.outer._local.indent += 1

            def __exit__(self, exc_type, exc, tb):
                self.outer._local.indent -= 1

        return _IndentCtx(self)

    def _suppressing_indent(self):
        """Contexto que incrementa indent pero oculta el nivel actual."""
        self._ensure_local_state()

        class _SuppressCtx:
            def __init__(self, outer):
                self.outer = outer

            def __enter__(self):
                self.outer._local.suppress += 1
                self.outer._local.indent += 1

            def __exit__(self, exc_type, exc, tb):
                self.outer._local.indent -= 1
                self.outer._local.suppress -= 1

        return _SuppressCtx(self)

    # ------------------------------------------------------------------
    # decoración automática
    def auto_indent(self, func=None):
        """Decorador que controla la indentación automática.

        Parameters
        ----------
        func : callable | None
            Función a decorar.
        """

        def decorator(inner_func):
            @functools.wraps(inner_func)
            def wrapper(*args, **kwargs):
                self._ensure_local_state()
                current_indent = self._local.indent
                ctx = self._suppressing_indent() if current_indent == 0 else self.indent()
                with ctx:
                    return inner_func(*args, **kwargs)

            return wrapper

        if func is None:
            return decorator
        return decorator(func)

    def auto_indent_methods(
        self,
        cls=None,
        *,
        include_private: bool = True,
        include_dunder: bool = True,
    ):
        """Decora todos los métodos de una clase con :meth:`auto_indent`.

        Parameters
        ----------
        cls : type | None
            Clase a decorar. Si es ``None`` retorna el decorador.
        include_private : bool, default True
            Cuando es False, ignora miembros cuyo nombre comienza con ``_``.
        include_dunder : bool, default True
            Cuando es False, ignora métodos ``__dunder__``.
        """

        def should_wrap(name: str) -> bool:
            if not include_dunder and name.startswith("__") and name.endswith("__"):
                return False
            if not include_private and name.startswith("_"):
                return False
            return True

        def wrap_class(target_cls):
            for name, attr in vars(target_cls).items():
                if not should_wrap(name):
                    continue

                descriptor = None
                if isinstance(attr, staticmethod):
                    descriptor = staticmethod
                    attr = attr.__func__
                elif isinstance(attr, classmethod):
                    descriptor = classmethod
                    attr = attr.__func__
                elif isinstance(attr, property):
                    # Extract all property methods
                    fget, fset, fdel = attr.fget, attr.fset, attr.fdel
                    
                    # Wrap each method if it exists
                    wrapped_fget = self.auto_indent(fget) if fget is not None else None
                    wrapped_fset = self.auto_indent(fset) if fset is not None else None
                    wrapped_fdel = self.auto_indent(fdel) if fdel is not None else None
                    
                    # Recreate property with wrapped methods and original docstring
                    wrapped = property(wrapped_fget, wrapped_fset, wrapped_fdel, attr.__doc__)
                    setattr(target_cls, name, wrapped)
                    continue

                if callable(attr):
                    wrapped = self.auto_indent(attr)
                    if descriptor is not None:
                        wrapped = descriptor(wrapped)
                    setattr(target_cls, name, wrapped)

            return target_cls

        if cls is None:
            return wrap_class
        return wrap_class(cls)

    # ------------------------------------------------------------------
    # logging helpers
    def _apply_indent(self, msg: str) -> str:
        self._ensure_local_state()
        level = max(self._local.indent - self._local.suppress, 0)
        indent = self.INDENT_STR * level
        return f"{indent}/> {msg}"

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(self._apply_indent(msg), *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.logger.info(self._apply_indent(msg), *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(self._apply_indent(msg), *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(self._apply_indent(msg), *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self.logger.critical(self._apply_indent(msg), *args, **kwargs)


if __name__ == "__main__":
    logger = HierLogger("test_logger")

    @logger.auto_indent
    def fun():
        logger.debug("fun()")
        fin()
        fin()
        main()

    @logger.auto_indent
    def fin():
        logger.debug("fin()")
        main()

    @logger.auto_indent
    def main():
        logger.debug("main()")

    logger.debug("msg-inicio")
    fin()
    fun()
    logger.debug("msg-fin")

    @logger.auto_indent_methods
    class Worker:
        def step_one(self):
            logger.debug("worker.step_one")
            self.step_two()

        def step_two(self):
            logger.debug("worker.step_two")

    Worker().step_one()
