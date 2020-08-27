
header-links: remove-links
	HEADERLIST="$(top_srcdir)/*.h $(top_srcdir)/*/include/*.h"; \
	for h in $$HEADERLIST; do \
	  BASENAME=`basename $$h`; \
	  test -r $$BASENAME || $(LN_S) $$h $$BASENAME; \
	done

remove-links: 
	rm -f *.h


all: all-am header-links

clean-local: remove-links
distclean-local: remove-links
